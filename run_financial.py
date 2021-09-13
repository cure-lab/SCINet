import os
import torch
from datetime import datetime
from experiments.exp_financial import Exp_financial
import argparse
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
### -------  dataset settings --------------
parser.add_argument('--dataset_name', type=str, default='exchange_rate', choices=['electricity', 'solar_AL', 'exchange_rate', 'traffic'])
parser.add_argument('--data', type=str, default='./datasets/exchange_rate.txt',
                    help='location of the data file')
parser.add_argument('--normalize', type=int, default=2)

### -------  device settings --------------
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')

### -------  input/output length settings --------------                                                                            
parser.add_argument('--window_size', type=int, default=168) # input size
parser.add_argument('--horizon', type=int, default=3)  # predication
parser.add_argument('--concat_len', type=int, default=165)
parser.add_argument('--single_step', type=int, default=0)
parser.add_argument('--single_step_output_One', type=int, default=0)
parser.add_argument('--lastWeight', type=float, default=1.0,help='Loss weight lambda')

### -------  training settings --------------  
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--num_nodes',type=int,default=8,help='number of nodes/variables')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=5e-3,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--lradj', type=int, default=6,help='adjust learning rate')
parser.add_argument('--save_path', type=str, default='./financial_checkpoints/')

### -------  model settings --------------  
parser.add_argument('--hidden-size', default=1.0, type=float, help='hidden channel of module')# H, EXPANSION RATE
parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
parser.add_argument('--kernel', default=5, type=int, help='kernel size')#k kernel size
parser.add_argument('--dilation', default=1, type=int, help='dilation')
parser.add_argument('--positionalEcoding', type = bool , default=False)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--stacks', type=int, default=1)
parser.add_argument('--model_name', type=str, default='SCNet')


args = parser.parse_args()

args.concat_len = args.window_size - args.horizon

if __name__ == '__main__':

    torch.manual_seed(4321)  # reproducible
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # Can change it to False --> default: False
    torch.backends.cudnn.enabled = True

    Exp=Exp_financial
    exp=Exp(args)

    if args.train:
        before_train = datetime.now().timestamp()
        print("===================Normal-Start=========================")
        normalize_statistic = exp.train()
        after_train = datetime.now().timestamp()
        print(f'Training took {(after_train - before_train) / 60} minutes')
        print("===================Normal-End=========================")
    
    else:
        data=exp._get_data()
        exp.model=exp.load_model(args.save_dir, epoch=args.epoch)
        before_evaluation = datetime.now().timestamp()
        rse, rae, correlation,rse_mid, rae_mid, correlation_mid=exp.test(data,data.test[0],data.test[1])
        after_evaluation = datetime.now().timestamp()
        print(
        '|Test_final rse {:5.4f} | Test_final rae {:5.4f} | Test_final corr   {:5.4f}'.format(
            rse, rae, correlation), flush=True)
        print(
        '|Test_mid rse {:5.4f} | Test_mid rae {:5.4f} | Test_mid corr  {:5.4f}'.format(
            rse_mid, rae_mid, correlation_mid), flush=True)
        print(f'Evaluation took {(after_evaluation - before_evaluation) / 60} minutes')





