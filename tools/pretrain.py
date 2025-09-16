
# tools/pretrain.py

import argparse
import os
import random
import warnings
import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import setup_env

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Unsupervised Pre-training")
    parser.add_argument("-f", "--exp_file", default=None, type=str, required=True, help="path to experiment file")
    parser.add_argument("-d", "--devices", default=1, type=int, help="number of devices (gpus)")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--local_rank", default=0, type=int)
    return parser

@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. This will turn on CUDNN deterministic setting, which can slow down training.")
    
    # The get_trainer method in our experiment file returns the PretrainTrainer
    trainer = exp.get_trainer(args)
    trainer.train()

if __name__ == "__main__":
    args = make_parser().parse_args()
    setup_env() # Sets up YOLOX environment
    exp = get_exp(args.exp_file)

    launch(
        main,
        args.devices,
        1, # num_machines
        "pretrain", # machine_rank
        backend="nccl",
        dist_url="auto",
        args=(exp, args),
    )
