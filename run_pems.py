import os
import torch
from datetime import datetime
from experiments.exp_pems import Exp_pems
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='SCINet on pems datasets')

### -------  dataset settings --------------
parser.add_argument('--dataset', type=str, default='PEMS08', choices=['PEMS03', 'PEMS04', 'PEMS07', 'PEMS08'])  #sometimes use: PeMS08
parser.add_argument('--norm_method', type=str, default='z_score')
parser.add_argument('--normtype', type=int, default=0)

### -------  device settings --------------
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--device', type=str, default='cuda:0')

### -------  input/output length settings --------------                                                                            
parser.add_argument('--window_size', type=int, default=12)
parser.add_argument('--horizon', type=int, default=12)
parser.add_argument('--concat_len', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)

parser.add_argument('--train_length', type=float, default=6)
parser.add_argument('--valid_length', type=float, default=2)
parser.add_argument('--test_length', type=float, default=2)

### -------  training settings --------------  
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--finetune', type=bool, default=False)
parser.add_argument('--validate_freq', type=int, default=1)

parser.add_argument('--epoch', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--optimizer', type=str, default='N') #
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--exponential_decay_step', type=int, default=5)
parser.add_argument('--decay_rate', type=float, default=0.5)

parser.add_argument('--lradj', type=int, default=1,help='adjust learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model_name', type=str, default='SCINet')

### -------  model settings --------------  
parser.add_argument('--hidden-size', default=0.0625, type=float, help='hidden channel scale of module')
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size for the first layer')
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type=bool , default = True)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=2)
parser.add_argument('--stacks', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_decoder_layer', type=int, default=1)
parser.add_argument('--RIN', type=bool, default=False)
parser.add_argument('--decompose', type=bool,default=False)


args = parser.parse_args()

if __name__ == '__main__':

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    Exp=Exp_pems
    exp=Exp(args)

    if args.evaluate:
        before_evaluation = datetime.now().timestamp()
        exp.test()
        after_evaluation = datetime.now().timestamp()
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')
    elif args.train or args.resume:
        before_train = datetime.now().timestamp()
        print("===================Normal-Start=========================")
        _, normalize_statistic = exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Normal-End=========================")

