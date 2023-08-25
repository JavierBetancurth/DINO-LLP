import argparse
import os
import uuid
from pathlib import Path

import main_dino_TPU
import submitit
import torch_xla.distributed.xla_multiprocessing as xmp

def parse_args():
    parser = argparse.ArgumentParser("Submitit for DINO", parents=[main_dino_TPU.get_args_parser()])
    parser.add_argument("--tpu_num_cores", default=8, type=int, help="Number of TPU cores to use")

    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, index):
        self.args.gpu = index
        main_dino.train_dino(self.args)

def main(index):
    args = parse_args()
    if args.output_dir == "":
        args.output_dir = f"/content/experiments/{index}"  # Change this to your desired directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.dist_url = f"tcp://localhost:{12345}"  # Change the port as needed

    xmp.spawn(Trainer(args), args=(), nprocs=1, start_method='fork')

if __name__ == "__main__":
    main(0)
