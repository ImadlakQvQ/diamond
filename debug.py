import sys
import runpy
import os

os.chdir('/home/xz/program/diamond')
sys.path.append('/home/xz/program/diamond/src')

args = 'python src/main.py --config-name debug env.train.id=BreakoutNoFrameskip-v4 common.devices=0'
args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')