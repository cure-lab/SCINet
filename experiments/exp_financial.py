import os
import math
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from metrics.Finantial_metics import MSE, MAE
from experiments.exp_basic import Exp_Basic
from data_process.financial_dataloader import DataLoaderH
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from utils.math_utils import smooth_l1_loss
from models.SCINet import SCINet
from models.SCINet_decompose import SCINet_decompose

class Exp_financial(Exp_Basic):
    def __init__(self, args):
        super(Exp_financial, self).__init__(args)
        if self.args.L1Loss:
            self.criterion = smooth_l1_loss
        else:
            self.criterion = nn.MSELoss(size_average=False).cuda()
        self.evaluateL2 = nn.MSELoss(size_average=False).cuda()
        self.evaluateL1 = nn.L1Loss(size_average=False).cuda()
        self.writer = SummaryWriter('.exp/run_financial/{}'.format(args.model_name))
    
    def _build_model(self):
        if self.args.dataset_name == 'electricity':
            self.input_dim = 321
            
        if self.args.dataset_name == 'solar_AL':
            self.input_dim = 137
            
        if self.args.dataset_name == 'exchange_rate':
            self.input_dim = 8
            
        if self.args.dataset_name == 'traffic':
            self.input_dim = 862
        
        if self.args.decompose:
            model = SCINet_decomp(
                output_len=self.args.horizon,
                input_len=self.args.window_size,
                input_dim=self.input_dim,
                hid_size=self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len=self.args.concat_len,
                groups=self.args.groups,
                kernel=self.args.kernel,
                dropout=self.args.dropout,
                single_step_output_One=self.args.single_step_output_One,
                positionalE=self.args.positionalEcoding,
                modified=True,
                RIN=self.args.RIN
            )
            
        else: 
            
            model = SCINet(
                output_len=self.args.horizon,
                input_len=self.args.window_size,
                input_dim=self.input_dim,
                hid_size=self.args.hidden_size,
                num_stacks=self.args.stacks,
                num_levels=self.args.levels,
                num_decoder_layer=self.args.num_decoder_layer,
                concat_len=self.args.concat_len,
                groups=self.args.groups,
                kernel=self.args.kernel,
                dropout=self.args.dropout,
                single_step_output_One=self.args.single_step_output_One,
                positionalE=self.args.positionalEcoding,
                modified=True,
                RIN=self.args.RIN
            )
        print(model)
        return model
    
    def _get_data(self):
        if self.args.dataset_name == 'electricity':
            self.args.data = './datasets/financial/electricity.txt'
            
        if self.args.dataset_name == 'solar_AL':
            self.args.data = './datasets/financial/solar_AL.txt'
            
        if self.args.dataset_name == 'exchange_rate':
            self.args.data = './datasets/financial/exchange_rate.txt'
            
        if self.args.dataset_name == 'traffic':
            self.args.data = './datasets/financial/traffic.txt'

        if self.args.long_term_forecast:
            return DataLoaderH(self.args.data, 0.7, 0.1, self.args.horizon, self.args.window_size, 4)
        else:
            return DataLoaderH(self.args.data, 0.6, 0.2, self.args.horizon, self.args.window_size, self.args.normalize)

    def _select_optimizer(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=1e-5)


    def train(self):

        best_val=10000000
        
        optim=self._select_optimizer()

        data=self._get_data()
        X=data.train[0]
        Y=data.train[1]
        save_path = os.path.join(self.args.save_path, self.args.model_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
        else:
            epoch_start = 0
            
        for epoch in range(epoch_start, self.args.epochs):
            epoch_start_time = time.time()
            iter = 0
            self.model.train()
            total_loss = 0
            n_samples = 0
            final_loss = 0
            min_loss = 0
            lr = adjust_learning_rate(optim, epoch, self.args)

            for tx, ty in data.get_batches(X, Y, self.args.batch_size, True):
                self.model.zero_grad()             #torch.Size([32, 168, 137])
                if self.args.stacks == 1:
                    forecast = self.model(tx)
                elif self.args.stacks == 2: 
                    forecast, res = self.model(tx)
                scale = data.scale.expand(forecast.size(0), self.args.horizon, data.m)
                bias = data.bias.expand(forecast.size(0), self.args.horizon, data.m)
                weight = torch.tensor(self.args.lastWeight).cuda() #used with multi-step

                if self.args.single_step: #single step
                    ty_last = ty[:, -1, :]
                    scale_last = data.scale.expand(forecast.size(0), data.m)
                    bias_last = data.bias.expand(forecast.size(0), data.m)
                    if self.args.normalize == 3:
                        loss_f = self.criterion(forecast[:, -1], ty_last)
                        if self.args.stacks == 2:
                            loss_m = self.criterion(res, ty)/res.shape[1] #average results

                    else:
                        loss_f = self.criterion(forecast[:, -1] * scale_last + bias_last, ty_last * scale_last + bias_last)
                        if self.args.stacks == 2:
                            loss_m = self.criterion(res * scale + bias, ty * scale + bias)/res.shape[1] #average results

                else:
                    if self.args.normalize == 3:
                        if self.args.lastWeight == 1.0:
                            loss_f = self.criterion(forecast, ty)
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res, ty)
                        else:
                            loss_f = self.criterion(forecast[:, :-1, :], ty[:, :-1, :] ) \
                                    + weight * self.criterion(forecast[:, -1:, :], ty[:, -1:, :] )
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res[:, :-1, :] , ty[:, :-1, :] ) \
                                        + weight * self.criterion(res[:, -1:, :], ty[:, -1:, :] )
                    else:
                        if self.args.lastWeight == 1.0:
                            loss_f = self.criterion(forecast * scale + bias, ty * scale + bias)
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res * scale + bias, ty * scale + bias)
                        else:
                            loss_f = self.criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                            ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                                + weight * self.criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                                        ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])
                            if self.args.stacks == 2:
                                loss_m = self.criterion(res[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                                ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                                    + weight * self.criterion(res[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                                            ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])
                loss = loss_f
                if self.args.stacks == 2:
                    loss += loss_m

                loss.backward()
                total_loss += loss.item()

                final_loss  += loss_f.item()
                if self.args.stacks == 2:
                    min_loss  += loss_m.item()
                n_samples += (forecast.size(0) * data.m)
                grad_norm = optim.step()

                if iter%100==0:
                    if self.args.stacks == 1:
                        print('iter:{:3d} | loss: {:.7f}'.format(iter, loss.item()/(forecast.size(0) * data.m)))
                    elif self.args.stacks == 2:
                        print('iter:{:3d} | loss: {:.7f}, loss_final: {:.7f}, loss_mid: {:.7f}'.format(iter, loss.item()/(forecast.size(0) * data.m),
                                loss_f.item()/(forecast.size(0) * data.m),loss_m.item()/(forecast.size(0) * data.m)))
                iter += 1
            if self.args.stacks == 1:
                val_loss, val_rae, val_corr, val_mse, val_mae = self.validate(data, data.valid[0],data.valid[1])
                test_loss, test_rae, test_corr, test_mse, test_mae = self.validate(data, data.test[0],data.test[1])      
            elif self.args.stacks == 2:
                val_loss, val_rae, val_corr, val_rse_mid, val_rae_mid, val_correlation_mid, val_mse, val_mae =self.validate(data, data.valid[0],data.valid[1])
                test_loss, test_rae, test_corr, test_rse_mid, test_rae_mid, test_correlation_mid, test_mse, test_mae = self.validate(data, data.test[0],data.test[1])

            self.writer.add_scalar('Train_loss_tatal', total_loss / n_samples, global_step=epoch)
            self.writer.add_scalar('Train_loss_Final', final_loss / n_samples, global_step=epoch)
            self.writer.add_scalar('Validation_final_rse', val_loss, global_step=epoch)
            self.writer.add_scalar('Validation_final_rae', val_rae, global_step=epoch)
            self.writer.add_scalar('Validation_final_corr', val_corr, global_step=epoch)
            self.writer.add_scalar('Test_final_rse', test_loss, global_step=epoch)
            self.writer.add_scalar('Test_final_rae', test_rae, global_step=epoch)
            self.writer.add_scalar('Test_final_corr', test_corr, global_step=epoch)
            if self.args.stacks == 2:
                self.writer.add_scalar('Train_loss_Mid', min_loss / n_samples, global_step=epoch)
                self.writer.add_scalar('Validation_mid_rse', val_rse_mid, global_step=epoch)
                self.writer.add_scalar('Validation_mid_rae', val_rae_mid, global_step=epoch)
                self.writer.add_scalar('Validation_mid_corr', val_correlation_mid, global_step=epoch)
                self.writer.add_scalar('Test_mid_rse', test_rse_mid, global_step=epoch)
                self.writer.add_scalar('Test_mid_rae', test_rae_mid, global_step=epoch)
                self.writer.add_scalar('Test_mid_corr', test_correlation_mid, global_step=epoch)

            print(
                '| EncoDeco: end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}| valid mse {:5.4f} | valid mae  {:5.4f}|'
                ' test rse {:5.4f} | test rae {:5.4f} | test corr  {:5.4f} | test mse {:5.4f} | test mae  {:5.4f}|'.format(
                    epoch, (time.time() - epoch_start_time), total_loss / n_samples, val_loss, val_rae, val_corr, val_mse, val_mae, test_loss, test_rae, test_corr, test_mse, test_mae), flush=True)
            
            if val_mse < best_val and self.args.long_term_forecast:
                save_model(epoch, lr, self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
                print('--------------| Best Val loss |--------------')
                best_val = val_mse
            elif val_loss < best_val and not self.args.long_term_forecast:
                save_model(epoch, lr, self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
                print('--------------| Best Val loss |--------------')
                best_val = val_loss

        return total_loss / n_samples

    def validate(self, data, X, Y, evaluate=False):
        self.model.eval()
        total_loss = 0
        total_loss_l1 = 0

        total_loss_mid = 0
        total_loss_l1_mid = 0
        n_samples = 0
        predict = None
        res_mid = None
        test = None

        forecast_set = []
        Mid_set = []
        target_set = []

        if evaluate:
            save_path = os.path.join(self.args.save_path, self.args.model_name)
            self.model = load_model(self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)[0]

        for X, Y in data.get_batches(X, Y, self.args.batch_size, False):
            with torch.no_grad():
                if self.args.stacks == 1:
                    forecast = self.model(X)
                elif self.args.stacks == 2:
                    forecast, res = self.model(X) #torch.Size([32, 3, 137])
            # only predict the last step
            true = Y[:, -1, :].squeeze()
            output = forecast[:,-1,:].squeeze()

            forecast_set.append(forecast)
            target_set.append(Y)
            if self.args.stacks == 2:
                Mid_set.append(res)

            if len(forecast.shape)==1:
                forecast = forecast.unsqueeze(dim=0)
                if self.args.stacks == 2:
                    res = res.unsqueeze(dim=0)
            if predict is None:
                predict = forecast[:,-1,:].squeeze()
                test = Y[:,-1,:].squeeze() #torch.Size([32, 3, 137])
                if self.args.stacks == 2:
                    res_mid = res[:,-1,:].squeeze()

            else:
                predict = torch.cat((predict, forecast[:,-1,:].squeeze()))
                test = torch.cat((test, Y[:, -1, :].squeeze()))
                if self.args.stacks == 2:
                    res_mid = torch.cat((res_mid, res[:,-1,:].squeeze()))
            
            scale = data.scale.expand(output.size(0),data.m)
            bias = data.bias.expand(output.size(0), data.m)
            if self.args.stacks == 2:
                output_res = res[:,-1,:].squeeze()

            total_loss += self.evaluateL2(output * scale + bias, true * scale+ bias).item()
            total_loss_l1 += self.evaluateL1(output * scale+ bias, true * scale+ bias).item()
            if self.args.stacks == 2:
                total_loss_mid += self.evaluateL2(output_res * scale+ bias, true * scale+ bias).item()
                total_loss_l1_mid += self.evaluateL1(output_res * scale+ bias, true * scale+ bias).item()

            n_samples += (output.size(0) * data.m)

        forecast_Norm = torch.cat(forecast_set, axis=0)
        target_Norm = torch.cat(target_set, axis=0)
        mse = MSE(forecast_Norm.cpu().numpy(), target_Norm.cpu().numpy())
        mae = MAE(forecast_Norm.cpu().numpy(), target_Norm.cpu().numpy())

        if self.args.stacks == 2:
            Mid_Norm = torch.cat(Mid_set, axis=0)

        rse_final_each = []
        rae_final_each = []
        corr_final_each = []
        Scale = data.scale.expand(forecast_Norm.size(0),data.m)
        bias = data.bias.expand(forecast_Norm.size(0),data.m)
        if not self.args.single_step: #single step
            for i in range(forecast_Norm.shape[1]): #get results of each step
                lossL2_F = self.evaluateL2(forecast_Norm[:,i,:] * Scale + bias, target_Norm[:,i,:] * Scale+ bias).item()
                lossL1_F = self.evaluateL1(forecast_Norm[:,i,:] * Scale+ bias, target_Norm[:,i,:] * Scale+ bias).item()
                if self.args.stacks == 2:
                    lossL2_M = self.evaluateL2(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
                    lossL1_M = self.evaluateL1(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
                rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0]/ data.m) / data.rse
                rae_F = (lossL1_F / forecast_Norm.shape[0]/ data.m) / data.rae
                rse_final_each.append(rse_F.item())
                rae_final_each.append(rae_F.item())

                pred = forecast_Norm[:,i,:].data.cpu().numpy()
                y_true = target_Norm[:,i,:].data.cpu().numpy()

                sig_p = pred.std(axis=0)
                sig_g = y_true.std(axis=0)
                m_p = pred.mean(axis=0)
                m_g = y_true.mean(axis=0)
                ind = (sig_p * sig_g != 0)
                corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
                corr = (corr[ind]).mean()
                corr_final_each.append(corr)

        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae
        if self.args.stacks == 2:
            rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
            rae_mid = (total_loss_l1_mid / n_samples) / data.rae

        # only calculate the last step for financial datasets.
        predict = forecast_Norm.cpu().numpy()[:,-1,:]
        Ytest = target_Norm.cpu().numpy()[:,-1,:]

        sigma_p = predict.std(axis=0)
        sigma_g = Ytest.std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_p * sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
        if self.args.stacks == 2:
            mid_pred = Mid_Norm.cpu().numpy()[:,-1,:]
            sigma_mid = mid_pred.std(axis=0)
            mean_mid = mid_pred.mean(axis=0)
            index_mid = (sigma_mid * sigma_g != 0)
            correlation_mid = ((mid_pred - mean_mid) * (Ytest - mean_g)).mean(axis=0) / (sigma_mid * sigma_g)
            correlation_mid = (correlation_mid[index_mid]).mean()

        print(
            '|valid_final mse {:5.4f} |valid_final mae {:5.4f} |valid_final rse {:5.4f} | valid_final rae {:5.4f} | valid_final corr  {:5.4f}'.format(mse,mae,
                rse, rae, correlation), flush=True)
        if self.args.stacks == 2:
            print(
            '|valid_final mse {:5.4f} |valid_final mae {:5.4f} |valid_mid rse {:5.4f} | valid_mid rae {:5.4f} | valid_mid corr  {:5.4f}'.format(mse,mae,
                rse_mid, rae_mid, correlation_mid), flush=True)

        if self.args.stacks == 1:
            return rse, rae, correlation, mse, mae
        if self.args.stacks == 2:
            return rse, rae, correlation, rse_mid, rae_mid, correlation_mid, mse, mae
