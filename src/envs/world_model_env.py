from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple
# from torchvision import models, transforms
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import numpy as np
from coroutines import coroutine
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]

# resnet = models.resnet18(pretrained=True)
# resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(),])

# def extract_hidden_features(image, model):
#     # 提取图像的特征
#     image = transform(image).unsqueeze(0)  # 增加 batch 维度
#     with torch.no_grad():
#         features = model(image)
#     return features.flatten().cpu().numpy()

# def cosine_similarity(vec1, vec2):
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    sim: bool
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Denoiser,
        rew_end_model: RewEndModel,
        data_loader: DataLoader,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
    ) -> None:
        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        self.rew_end_model = rew_end_model
        self.horizon = cfg.horizon
        self.sim = cfg.sim
        self.return_denoising_trajectory = return_denoising_trajectory
        self.num_envs = data_loader.batch_sampler.batch_size
        self.generator_init = self.make_generator_init(data_loader, cfg.num_batches_to_preload)

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        obs, act, (hx, cx) = self.generator_init.send(self.num_envs)            # self.num_envs = 32
        self.obs_buffer = obs
        self.act_buffer = act
        self.hx_rew_end = hx
        self.cx_rew_end = cx
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        return self.obs_buffer[:, -1], {}

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor) -> None:
        obs, act, (hx, cx) = self.generator_init.send(dead.sum().item())
        self.obs_buffer[dead] = obs
        self.act_buffer[dead] = act
        self.hx_rew_end[:, dead] = hx
        self.cx_rew_end[:, dead] = cx
        self.ep_len[dead] = 0

    @torch.no_grad()
    def step(self, act: torch.LongTensor) -> StepOutput:
        self.act_buffer[:, -1] = act

        next_obs, denoising_trajectory = self.predict_next_obs()            # next_obs: torch.Size([32, 3, 64, 64])  denoising_trajectory: List[torch.Size([32, 3, 64, 64])]。这最后一个就是next_obs
        rew, end = self.predict_rew_end(next_obs.unsqueeze(1))

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)          # 将最远的图换到最前面
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs                   # 更新新的obs

        dead = torch.logical_or(end, trunc)

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if dead.any():
            self.reset_dead(dead)
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]

        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(self) -> Tuple[Tensor, List[Tensor]]:

        # TODO 这个地方不直接使用所有的obs_buffer，怎么把关键帧抽出来，抽4帧 
        # torch.Size([32, 4， 3， 64，64])   self.act_buffer.shape  torch.Size([0，1，1，3])
        return self.sampler.sample(self.obs_buffer, self.act_buffer)

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits_rew, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        rew = Categorical(logits=logits_rew).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init(
        self,
        data_loader: DataLoader,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield
        data_iterator = iter(data_loader)

        while True:
            # Preload on device and burnin rew/end model
            obs_, act_, hx_, cx_ = [], [], [], []
            for _ in range(num_batches_to_preload):
                batch = next(data_iterator)
                obs = batch.obs.to(self.device)
                act = batch.act.to(self.device)
                with torch.no_grad():
                    *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])  # Burn-in of rew/end model
                assert hx.size(0) == cx.size(0) == 1
                obs_.extend(list(obs))
                act_.extend(list(act))
                hx_.extend(list(hx[0]))
                cx_.extend(list(cx[0]))

            # Yield new initial conditions for dead envs
            c = 0
            while c + num_dead <= len(obs_):
                obs = torch.stack(obs_[c : c + num_dead])
                act = torch.stack(act_[c : c + num_dead])
                hx = torch.stack(hx_[c : c + num_dead]).unsqueeze(0)
                cx = torch.stack(cx_[c : c + num_dead]).unsqueeze(0)
                c += num_dead
                num_dead = yield obs, act, (hx, cx)
