import os
import time

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from experiments.exp_basic import Exp_Basic
from data_process.forecast_dataloader import ForecastDataset,ForecastTestDataset, de_normalized
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from torch.utils.tensorboard import SummaryWriter
from utils.math_utils import evaluate, creatMask
from models.SCINet import SCINet
from models.SCINet_decompose import SCINet_decompose

class Exp_pems(Exp_Basic):
    def __init__(self, args):
        super(Exp_pems, self).__init__(args)
        self.result_file = os.path.join('exp/pems_checkpoint', self.args.dataset, 'checkpoints')
        self.result_test_file = os.path.join('exp/pems_checkpoint', args.dataset, 'test')
        self.result_train_file = os.path.join('exp/pems_checkpoint', args.dataset, 'train')

    def _build_model(self):
        if self.args.dataset == 'PEMS03':
            self.input_dim = 358
        elif self.args.dataset == 'PEMS04':
            self.input_dim = 307
        elif self.args.dataset == 'PEMS07':
            self.input_dim = 883
        elif self.args.dataset == 'PEMS08':
            self.input_dim = 170
        if self.args.decompose:
            model = SCINet_decomp(
                output_len=self.args.horizon,
                input_len=self.args.window_size,
                input_dim=self.input_dim,
                hid_size = self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len = self.args.concat_len,
                groups = self.args.groups,
                kernel = self.args.kernel,
                dropout = self.args.dropout,
                single_step_output_One = self.args.single_step_output_One,
                positionalE = self.args.positionalEcoding,
                modified = True,
                RIN=self.args.RIN
            )
        else:
            model = SCINet(
                output_len=self.args.horizon,
                input_len=self.args.window_size,
                input_dim=self.input_dim,
                hid_size = self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len = self.args.concat_len,
                groups = self.args.groups,
                kernel = self.args.kernel,
                dropout = self.args.dropout,
                single_step_output_One = self.args.single_step_output_One,
                positionalE = self.args.positionalEcoding,
                modified = True,
                RIN=self.args.RIN
            )

        print(model)
        return model

    def _get_data(self):
        data_file = os.path.join('./datasets/PEMS', self.args.dataset + '.npz')
        print('data file:',data_file)
        data = np.load(data_file,allow_pickle=True)
        data = data['data'][:,:,0]
        train_ratio = self.args.train_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        valid_ratio = self.args.valid_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        test_ratio = 1 - train_ratio - valid_ratio
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        if len(train_data) == 0:
            raise Exception('Cannot organize enough training data')
        if len(valid_data) == 0:
            raise Exception('Cannot organize enough validation data')
        if len(test_data) == 0:
            raise Exception('Cannot organize enough test data')
        if self.args.normtype == 0: # we follow StemGNN and other related works for somewhat fair comparison (orz..), but we strongly suggest use self.args.normtype==2!!!
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_mean = np.mean(valid_data, axis=0)
            val_std = np.std(valid_data, axis=0)
            val_normalize_statistic = {"mean": val_mean.tolist(), "std": val_std.tolist()}
            test_mean = np.mean(test_data, axis=0)
            test_std = np.std(test_data, axis=0)
            test_normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}
        elif self.args.normtype == 1:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            train_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            val_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            test_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
        else:
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            test_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        train_set = ForecastDataset(train_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                normalize_method=self.args.norm_method, norm_statistic=train_normalize_statistic)
        valid_set = ForecastDataset(valid_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                    normalize_method=self.args.norm_method, norm_statistic=val_normalize_statistic)
        test_set = ForecastTestDataset(test_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                    normalize_method=self.args.norm_method, norm_statistic=test_normalize_statistic)
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, drop_last=False, shuffle=True,
                                            num_workers=1)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=1)
        node_cnt = train_data.shape[1]
        return test_loader, train_loader, valid_loader,node_cnt,test_normalize_statistic,val_normalize_statistic

    def _select_optimizer(self):
        if self.args.optimizer == 'RMSProp':
            my_optim = torch.optim.RMSprop(params=self.model.parameters(), lr=self.args.lr, eps=1e-08)
        else:
            my_optim = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)
        return my_optim

    def inference(self, model, dataloader, node_cnt, window_size, horizon):
        forecast_set = []
        Mid_set = []
        target_set = []
        input_set = []
        self.model.eval()
        with torch.no_grad():
            for i, (inputs, target) in enumerate(dataloader):
                inputs = inputs.cuda()
                target = target.cuda()
                input_set.append(inputs.detach().cpu().numpy())
                step = 0
                forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
                Mid_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)
                while step < horizon:
                    if self.args.stacks == 1:
                        forecast_result = self.model(inputs)
                    elif self.args.stacks == 2:
                        forecast_result, Mid_result = self.model(inputs)

                    len_model_output = forecast_result.size()[1]
                    if len_model_output == 0:
                        raise Exception('Get blank inference result')
                    inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                    :].clone()
                    inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                    forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                        forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                    if self.args.stacks == 2:
                        Mid_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                            Mid_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()

                    step += min(horizon - step, len_model_output)
                forecast_set.append(forecast_steps)
                target_set.append(target.detach().cpu().numpy())
                if self.args.stacks == 2:
                    Mid_set.append(Mid_steps)

                result_save = np.concatenate(forecast_set, axis=0)
                target_save = np.concatenate(target_set, axis=0)

        if self.args.stacks == 1:
            return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0), np.concatenate(input_set, axis=0)

        elif self.args.stacks == 2:
            return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0),np.concatenate(Mid_set, axis=0), np.concatenate(input_set, axis=0)

    def validate(self, model, epoch, forecast_loss, dataloader, normalize_method, statistic,
                node_cnt, window_size, horizon, writer,
                result_file=None,test=False):
        #start = datetime.now()
        print("===================Validate Normal=========================")
        if self.args.stacks == 1:
            forecast_norm, target_norm, input_norm = self.inference(model, dataloader, 
                                    node_cnt, window_size, horizon)
        elif self.args.stacks == 2:
            forecast_norm, target_norm, mid_norm, input_norm = self.inference(model, dataloader, 
                                            node_cnt, window_size, horizon)
        if normalize_method and statistic:
            forecast = de_normalized(forecast_norm, normalize_method, statistic)
            target = de_normalized(target_norm, normalize_method, statistic)
            input = de_normalized(input_norm, normalize_method, statistic)
            if self.args.stacks == 2:
                mid = de_normalized(mid_norm, normalize_method, statistic)
        else:
            forecast, target, input = forecast_norm, target_norm, input_norm
            if self.args.stacks == 2:
                mid = mid_norm

        beta = 0.1
        forecast_norm = torch.from_numpy(forecast_norm).float()
        target_norm = torch.from_numpy(target_norm).float()
        if self.args.stacks == 1:
            loss = forecast_loss(forecast_norm, target_norm)

        elif self.args.stacks == 2:
            mid_norm = torch.from_numpy(mid_norm).float()

            loss = forecast_loss(forecast_norm, target_norm) + forecast_loss(mid_norm, target_norm)
            loss_F = forecast_loss(forecast_norm, target_norm)
            loss_M = forecast_loss(mid_norm, target_norm)

        score = evaluate(target, forecast)
        score_final_detail = evaluate(target, forecast,by_step=True)
        print('by each step: MAPE & MAE & RMSE',score_final_detail)
        if self.args.stacks == 2:
            score1 = evaluate(target, mid)
        #end = datetime.now()

        if writer:
            if test:
                print(f'TEST: RAW : MAE {score[1]:7.2f};MAPE {score[0]:7.2f}; RMSE {score[2]:7.2f}.')
                writer.add_scalar('Test MAE_final', score[1], global_step=epoch)
                writer.add_scalar('Test RMSE_final', score[2], global_step=epoch)
                if self.args.stacks == 2:
                    print(f'TEST: RAW-Mid : MAE {score1[1]:7.2f}; MAPE {score[0]:7.2f}; RMSE {score1[2]:7.2f}.')
                    writer.add_scalar('Test MAE_Mid', score1[1], global_step=epoch)
                    writer.add_scalar('Test RMSE_Mid', score1[2], global_step=epoch)
                    writer.add_scalar('Test Loss_final', loss_F, global_step=epoch)
                    writer.add_scalar('Test Loss_Mid', loss_M, global_step=epoch)

            else:
                print(f'VAL: RAW : MAE {score[1]:7.2f}; RMSE {score[2]:7.2f}.')
                writer.add_scalar('VAL MAE_final', score[1], global_step=epoch)
                writer.add_scalar('VAL RMSE_final', score[2], global_step=epoch)

                if self.args.stacks == 2:
                    print(f'VAL: RAW-Mid : MAE {score1[1]:7.2f}; RMSE {score1[2]:7.2f}.')
                    writer.add_scalar('VAL MAE_Mid', score1[1], global_step=epoch)
                    writer.add_scalar('VAL RMSE_Mid', score1[2], global_step=epoch)
                    writer.add_scalar('VAL Loss_final', loss_F, global_step=epoch)
                    writer.add_scalar('VAL Loss_Mid', loss_M, global_step=epoch)

        if result_file:
            if not os.path.exists(result_file):
                os.makedirs(result_file)
            step_to_print = 0
            forcasting_2d = forecast[:, step_to_print, :]
            forcasting_2d_target = target[:, step_to_print, :]

            np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
            np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
            np.savetxt(f'{result_file}/predict_abs_error.csv',
                    np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
            np.savetxt(f'{result_file}/predict_ape.csv',
                    np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

        return dict(mae=score[1], mape=score[0], rmse=score[2])


    def train(self):
        my_optim=self._select_optimizer()
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=self.args.decay_rate)
        test_loader, train_loader, valid_loader,node_cnt,test_normalize_statistic,val_normalize_statistic=self._get_data()
        forecast_loss = nn.L1Loss().cuda()
        best_validate_mae = np.inf
        best_test_mae = np.inf
        validate_score_non_decrease_count = 0
        writer = SummaryWriter('exp/run_PEMS/{}_scinet'.format(self.args.model_name))
        
        performance_metrics = {}

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, self.result_file, model_name=self.args.dataset, horizon=self.args.horizon)
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args.epoch):
            lr = adjust_learning_rate(my_optim, epoch, self.args)
            epoch_start_time = time.time()
            self.model.train()
            loss_total = 0
            loss_total_F = 0
            loss_total_M = 0
            cnt = 0
            for i, (inputs, target) in enumerate(train_loader):
                inputs = inputs.cuda()  # torch.Size([32, 12, 228])
                target = target.cuda()  # torch.Size([32, 3, 228])
                self.model.zero_grad()
                if self.args.stacks == 1:
                    forecast = self.model(inputs)
                    loss = forecast_loss(forecast, target)
                elif self.args.stacks == 2:
                    forecast, res = self.model(inputs)
                    loss = forecast_loss(forecast, target) + forecast_loss(res, target)
                    loss_M = forecast_loss(res, target)
                    loss_F = forecast_loss(forecast, target)
                
                cnt += 1
                loss.backward()
                my_optim.step()
                loss_total += float(loss)
                if self.args.stacks == 2:
                    loss_total_F  += float(loss_F)
                    loss_total_M  += float(loss_M)
            if self.args.stacks == 1:
                print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f} '.format(epoch, (
                    time.time() - epoch_start_time), loss_total / cnt))
            elif self.args.stacks == 2:
                print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}, loss_F {:5.4f}, loss_M {:5.4f}  '.format(epoch, (
                    time.time() - epoch_start_time), loss_total / cnt, loss_total_F / cnt, loss_total_M / cnt))

            writer.add_scalar('Train_loss_tatal', loss_total / cnt, global_step=epoch)
            if self.args.stacks == 2:
                writer.add_scalar('Train_loss_Mid', loss_total_F / cnt, global_step=epoch)
                writer.add_scalar('Train_loss_Final', loss_total_M / cnt, global_step=epoch)

            if (epoch+1) % self.args.exponential_decay_step == 0:
                my_lr_scheduler.step()
            if (epoch + 1) % self.args.validate_freq == 0:
                is_best_for_now = False
                print('------ validate on data: VALIDATE ------')
                performance_metrics = self.validate(self.model, epoch, forecast_loss, valid_loader, self.args.norm_method, val_normalize_statistic,
                            node_cnt, self.args.window_size, self.args.horizon,
                            writer, result_file=None, test=False)
                test_metrics = self.validate(self.model, epoch,  forecast_loss, test_loader, self.args.norm_method, test_normalize_statistic,
                            node_cnt, self.args.window_size, self.args.horizon,
                            writer, result_file=None, test=True)
                if best_validate_mae > performance_metrics['mae']:
                    best_validate_mae = performance_metrics['mae']
                    is_best_for_now = True
                    validate_score_non_decrease_count = 0
                    print('got best validation result:',performance_metrics, test_metrics)
                else:
                    validate_score_non_decrease_count += 1
                if best_test_mae > test_metrics['mae']:
                    best_test_mae = test_metrics['mae']
                    print('got best test result:', test_metrics)
                    
                # save model
                if is_best_for_now:
                    save_model(epoch, lr, model=self.model, model_dir=self.result_file, model_name=self.args.dataset, horizon=self.args.horizon)
                    print('saved model!')
            # early stop
            if self.args.early_stop and validate_score_non_decrease_count >= self.args.early_stop_step:
                break
        return performance_metrics, test_normalize_statistic

    def test(self, epoch=None):
        data_file = os.path.join('./datasets/PEMS', self.args.dataset + '.npz')
        data = np.load(data_file,allow_pickle=True)
        data = data['data'][:,:,0]
        train_ratio = self.args.train_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        valid_ratio = self.args.valid_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        result_train_file=self.result_train_file
        result_test_file=self.result_test_file

        test_mean = np.mean(test_data, axis=0)
        test_std = np.std(test_data, axis=0)
        normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}

        forecast_loss = nn.L1Loss().cuda() #smooth_l1_loss #nn.MSELoss(reduction='mean').cuda()
        model = load_model(self.model, self.result_file, model_name=self.args.dataset, horizon=self.args.horizon)
        node_cnt = test_data.shape[1]
        test_set = ForecastTestDataset(test_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                normalize_method=self.args.norm_method, norm_statistic=normalize_statistic)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size*10, drop_last=False,
                                            shuffle=False, num_workers=0)
        performance_metrics = self.validate(model = model, epoch = 100, forecast_loss = forecast_loss, dataloader = test_loader, normalize_method = self.args.norm_method, statistic = normalize_statistic,
                        node_cnt = node_cnt, window_size = self.args.window_size, horizon =self.args.horizon,
                        result_file=result_test_file, writer = None, test=True)
        mae, rmse, mape = performance_metrics['mae'], performance_metrics['rmse'], performance_metrics['mape']
        print('Performance on test set: | MAE: {:5.2f} | MAPE: {:5.2f} | RMSE: {:5.4f}'.format(mae, mape, rmse))
