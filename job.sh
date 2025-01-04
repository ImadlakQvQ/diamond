#!/bin/bash
#SBATCH --account=def-bboulet     # set account
#SBATCH --gpus-per-node=2         # Number of GPU(s) per node
#SBATCH --output=log/%j.out      # log 保存地址
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=80gb               # memory per node
#SBATCH --time=3-02:01

module load python/3.10.13 gcc cuda opencv/4.10.0
source /home/imadlak/projects/def-bboulet/imadlak/program/diamond/env/bin/activate
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/home/imadlak/projects/def-bboulet/imadlak/program/diamond_/diamond:$PYTHONPATH
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.devices=0
