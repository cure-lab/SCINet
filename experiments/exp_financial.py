from experiments.exp_basic import Exp_Basic
# from models.model import Informer, InformerStack
from data_process.financial_dataloader import DataLoaderH
from utils.tools import EarlyStopping, adjust_learning_rate
from metrics.ETTh_metrics import metric
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.math_utils import smooth_l1_loss
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import os
import time

import warnings
warnings.filterwarnings('ignore')

from models.SCINet import SCINet, SCINetEncoder

import math



class Exp_financial(Exp_Basic):
    def __init__(self, args):
        super(Exp_financial, self).__init__(args)
        if self.args.L1Loss:
            self.criterion = smooth_l1_loss #nn.L1Loss(size_average=False).to(device)  nn.L1Loss().to(args.device)
        #    criterion =  nn.L1Loss().to(args.device)
        else:
            self.criterion = nn.MSELoss(size_average=False).to(self.args.device)
        self.evaluateL2 = nn.MSELoss(size_average=False).to(self.args.device)
        self.evaluateL1 = nn.L1Loss(size_average=False).to(self.args.device)
        self.writer = SummaryWriter('./run_financial/{}'.format(args.model_name))
    
    def _build_model(self):
        model = SCINet(self.args, output_len = self.args.horizon, input_len=self.args.window_size, input_dim = self.args.num_nodes,
                num_layers = 3, concat_len= self.args.num_concat)
        #model = model.to(device)
        return model
    
    def _get_data(self):
        if self.args.data == './datasets/financial/electricity.txt':
            self.args.num_nodes = 321
            self.args.dataset_name = 'electricity'

        if self.args.data == './datasets/financial/solar_AL.txt':
            self.args.num_nodes = 137
            self.args.dataset_name = 'solar_AL'

        if self.args.data == './datasets/financial/exchange_rate.txt':
            self.args.num_nodes = 8
            self.args.dataset_name = 'exchange_rate'

        if self.args.data == './datasets/financial/traffic.txt':
            self.args.num_nodes = 862
            self.args.dataset_name = 'traffic'

        print('dataset {}, the channel size is {}'.format(self.args.data, self.args.num_nodes))

        return DataLoaderH(self.args.data, 0.6, 0.2, self.args.device, self.args.horizon, self.args.window_size, self.args.normalize)

    def _select_optimizer(self):
        return torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=1e-5)

    def save_model(self, model, model_dir, epoch=None):
        if model_dir is None:
            return
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        epoch = str(epoch) if epoch else ''
        file_name = os.path.join(model_dir, epoch + 'financial_best.pt')
        with open(file_name, 'wb') as f:
            torch.save(model, f)
    def load_model(self, model_dir, epoch=None):
        if not model_dir:
            return
        epoch = str(epoch) if epoch else ''
        file_name = os.path.join(model_dir, epoch + 'financial_best.pt')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(file_name):
            return
        with open(file_name, 'rb') as f:
            model = torch.load(f)

    def train(self):
        # epoch, data, X, Y, model, criterion, optim, batch_size
        

        best_val=10000000
        
        optim=self._select_optimizer()

        data=self._get_data()
        X=data.train[0]
        Y=data.train[1]

        for epoch in range(1, self.args.epochs+1):
            epoch_start_time = time.time()
            iter = 0
            self.model.train()
            total_loss = 0
            n_samples = 0
            final_loss = 0
            min_loss = 0
            adjust_learning_rate(optim, epoch, self.args)

            for tx, ty in data.get_batches(X, Y, self.args.batch_size, True):
                self.model.zero_grad()             #torch.Size([32, 168, 137])

                forecast, res = self.model(tx)
                # forecast = torch.squeeze(forecast)
                scale = data.scale.expand(forecast.size(0), self.args.horizon, data.m)
                bias = data.bias.expand(forecast.size(0), self.args.horizon, data.m)
                weight = torch.tensor(self.args.lastWeight).to(self.args.device)

                # if args.normalize == 3:
                #     # loss = criterion(forecast, ty) + criterion(res, ty)
                # else:
                    # loss = criterion(forecast * scale + bias, ty * scale + bias) + criterion(res * scale + bias, ty * scale + bias)



                if self.args.normalize == 3:
                    if self.args.lastWeight == 1.0:
                        loss_f = self.criterion(forecast, ty)
                        loss_m = self.criterion(res, ty)
                    else:

                        loss_f = self.criterion(forecast[:, :-1, :] ,
                                        ty[:, :-1, :] ) \
                                + weight * self.criterion(forecast[:, -1:, :],
                                                    ty[:, -1:, :] )
                        loss_m = self.criterion(res[:, :-1, :] ,
                                        ty[:, :-1, :] ) \
                                + weight * self.criterion(res[:, -1:, :],
                                                    ty[:, -1:, :] )
                else:
                    if self.args.lastWeight == 1.0:
                        loss_f = self.criterion(forecast * scale + bias, ty * scale + bias)
                        loss_m = self.criterion(res * scale + bias, ty * scale + bias)
                    else:

                        loss_f = self.criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                        ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                            + weight * self.criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                                    ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])
                        # print(self.criterion(forecast[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]),weight * self.criterion(forecast[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :]))
                        loss_m = self.criterion(res[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :],
                                        ty[:, :-1, :] * scale[:, :-1, :] + bias[:, :-1, :]) \
                            + weight * self.criterion(res[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :],
                                                    ty[:, -1:, :] * scale[:, -1:, :] + bias[:, -1:, :])

                loss = loss_f+loss_m

                loss.backward()
                total_loss += loss.item()

                final_loss  += loss_f.item()
                min_loss  += loss_m.item()
                n_samples += (forecast.size(0) * data.m)
                grad_norm = optim.step()

                if iter%100==0:
                    print('iter:{:3d} | loss: {:.7f}, loss_final: {:.7f}, loss_mid: {:.7f}'.format(iter,loss.item()/(forecast.size(0) * data.m),
                                                                                                loss_f.item()/(forecast.size(0) * data.m),loss_m.item()/(forecast.size(0) * data.m)))
                iter += 1

            val_loss, val_rae, val_corr, val_rse_mid, val_rae_mid, val_correlation_mid=self.validate(data, data.valid[0],data.valid[1])
            test_loss, test_rae, test_corr, test_rse_mid, test_rae_mid, test_correlation_mid= self.test(data, data.test[0],data.test[1])

            self.writer.add_scalar('Validation_final_rse', val_loss, global_step=epoch)
            self.writer.add_scalar('Validation_final_rae', val_rae, global_step=epoch)
            self.writer.add_scalar('Validation_final_corr', val_corr, global_step=epoch)

            self.writer.add_scalar('Validation_mid_rse', val_rse_mid, global_step=epoch)
            self.writer.add_scalar('Validation_mid_rae', val_rae_mid, global_step=epoch)
            self.writer.add_scalar('Validation_mid_corr', val_correlation_mid, global_step=epoch)

            self.writer.add_scalar('Test_final_rse', test_loss, global_step=epoch)
            self.writer.add_scalar('Test_final_rae', test_rae, global_step=epoch)
            self.writer.add_scalar('Test_final_corr', test_corr, global_step=epoch)

            self.writer.add_scalar('Test_mid_rse', test_rse_mid, global_step=epoch)
            self.writer.add_scalar('Test_mid_rae', test_rae_mid, global_step=epoch)
            self.writer.add_scalar('Test_mid_corr', test_correlation_mid, global_step=epoch)

            self.writer.add_scalar('Train_loss_tatal', total_loss / n_samples, global_step=epoch)
            self.writer.add_scalar('Train_loss_Mid', min_loss / n_samples, global_step=epoch)
            self.writer.add_scalar('Train_loss_Final', final_loss / n_samples, global_step=epoch)
            print(
                '| EncoDeco: end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}|'
                ' test rse {:5.4f} | test rae {:5.4f} | test corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), total_loss / n_samples, val_loss, val_rae, val_corr, test_loss, test_rae, test_corr), flush=True)
            if val_loss<best_val:
                self.save_model(self.model,self.args.save_path,epoch=epoch)
                print('--------------| Best Val loss |--------------')
        return total_loss / n_samples

    def test(self, data, X, Y):
        # data=self._get_data()
        # X=data.test[0]
        # Y=data.test[1]
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

        for X, Y in data.get_batches(X, Y, self.args.batch_size*10, False):
            # print('0')
            # X = torch.unsqueeze(X,dim=1)
            # X = X.transpose(2,3)
            with torch.no_grad():
                forecast, res = self.model(X) #torch.Size([32, 3, 137])
            # forecast = torch.squeeze(forecast)
            # res = torch.squeeze(res)
            true = Y[:, -1, :].squeeze()

            forecast_set.append(forecast)
            Mid_set.append(res)
            target_set.append(Y)

            if len(forecast.shape)==1:
                forecast = forecast.unsqueeze(dim=0)
                res = res.unsqueeze(dim=0)
            if predict is None:
                predict = forecast[:, -1, :].squeeze()
                res_mid = res[:, -1, :].squeeze()
                test = Y[:, -1, :].squeeze()  # torch.Size([32, 3, 137])

            else:
                predict = torch.cat((predict, forecast[:, -1, :].squeeze()))
                res_mid = torch.cat((res_mid, res[:, -1, :].squeeze()))
                test = torch.cat((test, Y[:, -1, :].squeeze()))
            output = forecast[:, -1, :].squeeze()
            output_res = res[:, -1, :].squeeze()
            scale = data.scale.expand(output.size(0), data.m)
            bias = data.bias.expand(output.size(0), data.m)

            total_loss += self.evaluateL2(output * scale + bias, true * scale+ bias).item()
            total_loss_l1 += self.evaluateL1(output * scale + bias, true * scale+ bias).item()
            total_loss_mid += self.evaluateL2(output_res * scale + bias, true * scale+ bias).item()
            total_loss_l1_mid += self.evaluateL1(output_res * scale + bias, true * scale+ bias).item()

            n_samples += (output.size(0) * data.m)

        forecast_Norm = torch.cat(forecast_set, axis=0)
        target_Norm = torch.cat(target_set, axis=0)
        Mid_Norm = torch.cat(Mid_set, axis=0)

        rse_final_each = []
        rae_final_each = []
        corr_final_each = []
        Scale = data.scale.expand(forecast_Norm.size(0), data.m)
        bias = data.bias.expand(forecast_Norm.size(0), data.m)
        for i in range(forecast_Norm.shape[1]):
            lossL2_F = self.evaluateL2(forecast_Norm[:, i, :] * Scale + bias, target_Norm[:, i, :] * Scale + bias).item()
            lossL1_F = self.evaluateL1(forecast_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
            lossL2_M = self.evaluateL2(Mid_Norm[:, i, :] * Scale + bias, target_Norm[:, i, :] * Scale+ bias).item()
            lossL1_M = self.evaluateL1(Mid_Norm[:, i, :] * Scale + bias, target_Norm[:, i, :] * Scale+ bias).item()
            rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0] / data.m) / data.rse
            rae_F = (lossL1_F / forecast_Norm.shape[0] / data.m) / data.rae
            rse_final_each.append(rse_F.item())
            rae_final_each.append(rae_F.item())

            pred = forecast_Norm[:, i, :].data.cpu().numpy()
            y_true = target_Norm[:, i, :].data.cpu().numpy()

            sig_p = (pred).std(axis=0)
            sig_g = (y_true).std(axis=0)
            m_p = pred.mean(axis=0)
            m_g = y_true.mean(axis=0)
            ind = (sig_g != 0)
            corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
            corr = (corr[ind]).mean()
            corr_final_each.append(corr)

        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae

        rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
        rae_mid = (total_loss_l1_mid / n_samples) / data.rae

        predict = forecast_Norm.cpu().numpy()
        Ytest = target_Norm.cpu().numpy()

        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()

        mid_pred = Mid_Norm.cpu().numpy()
        sigma_p = (mid_pred).std(axis=0)
        mean_p = mid_pred.mean(axis=0)
        correlation_mid = ((mid_pred - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation_mid = (correlation_mid[index]).mean()

        print(
            '|Test_final rse {:5.4f} | Test_final rae {:5.4f} | Test_final corr   {:5.4f}'.format(
                rse, rae, correlation), flush=True)

        print(
            '|Test_mid rse {:5.4f} | Test_mid rae {:5.4f} | Test_mid corr  {:5.4f}'.format(
                rse_mid, rae_mid, correlation_mid), flush=True)
        # if epoch%4==0:
        #     save_model(model, result_file,epoch=epoch)
        return rse, rae, correlation, rse_mid, rae_mid, correlation_mid

    def validate(self,data,X,Y):
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

        for X, Y in data.get_batches(X, Y, self.args.batch_size*200, False):
            # print('0')
            # X = torch.unsqueeze(X,dim=1)
            # X = X.transpose(2,3)
            with torch.no_grad():
                forecast, res = self.model(X) #torch.Size([32, 3, 137])
            # forecast = torch.squeeze(forecast)
            # res = torch.squeeze(res)
            true = Y[:, -1, :].squeeze()


            forecast_set.append(forecast)
            Mid_set.append(res)
            target_set.append(Y)

            if len(forecast.shape)==1:
                forecast = forecast.unsqueeze(dim=0)
                res = res.unsqueeze(dim=0)
            if predict is None:
                predict = forecast[:,-1,:].squeeze()
                res_mid = res[:,-1,:].squeeze()
                test = Y[:,-1,:].squeeze() #torch.Size([32, 3, 137])

            else:
                predict = torch.cat((predict, forecast[:,-1,:].squeeze()))
                res_mid = torch.cat((res_mid, res[:,-1,:].squeeze()))
                test = torch.cat((test, Y[:, -1, :].squeeze()))
            output = forecast[:,-1,:].squeeze()
            output_res = res[:,-1,:].squeeze()
            scale = data.scale.expand(output.size(0),data.m)
            bias = data.bias.expand(output.size(0), data.m)


            total_loss += self.evaluateL2(output * scale + bias, true * scale+ bias).item()
            total_loss_l1 += self.evaluateL1(output * scale+ bias, true * scale+ bias).item()
            total_loss_mid += self.evaluateL2(output_res * scale+ bias, true * scale+ bias).item()
            total_loss_l1_mid += self.evaluateL1(output_res * scale+ bias, true * scale+ bias).item()

            n_samples += (output.size(0) * data.m)

        forecast_Norm = torch.cat(forecast_set, axis=0)
        target_Norm = torch.cat(target_set, axis=0)
        Mid_Norm = torch.cat(Mid_set, axis=0)


        rse_final_each = []
        rae_final_each = []
        corr_final_each = []
        Scale = data.scale.expand(forecast_Norm.size(0),data.m)
        bias = data.bias.expand(forecast_Norm.size(0),data.m)
        for i in range(forecast_Norm.shape[1]):
            lossL2_F = self.evaluateL2(forecast_Norm[:,i,:] * Scale + bias, target_Norm[:,i,:] * Scale+ bias).item()
            lossL1_F = self.evaluateL1(forecast_Norm[:,i,:] * Scale+ bias, target_Norm[:,i,:] * Scale+ bias).item()
            lossL2_M = self.evaluateL2(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
            lossL1_M = self.evaluateL1(Mid_Norm[:, i, :] * Scale+ bias, target_Norm[:, i, :] * Scale+ bias).item()
            rse_F = math.sqrt(lossL2_F / forecast_Norm.shape[0]/ data.m) / data.rse
            rae_F = (lossL1_F / forecast_Norm.shape[0]/ data.m) / data.rae
            rse_final_each.append(rse_F.item())
            rae_final_each.append(rae_F.item())

            pred = forecast_Norm[:,i,:].data.cpu().numpy()
            y_true = target_Norm[:,i,:].data.cpu().numpy()

            sig_p = (pred).std(axis=0)
            sig_g = (y_true).std(axis=0)
            m_p = pred.mean(axis=0)
            m_g = y_true.mean(axis=0)
            ind = (sig_g != 0)
            corr = ((pred - m_p) * (y_true - m_g)).mean(axis=0) / (sig_p * sig_g)
            corr = (corr[ind]).mean()
            corr_final_each.append(corr)

        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = (total_loss_l1 / n_samples) / data.rae

        rse_mid = math.sqrt(total_loss_mid / n_samples) / data.rse
        rae_mid = (total_loss_l1_mid / n_samples) / data.rae

        predict = forecast_Norm.cpu().numpy()
        Ytest = target_Norm.cpu().numpy()

        sigma_p = (predict).std(axis=0)
        sigma_g = (Ytest).std(axis=0)
        mean_p = predict.mean(axis=0)
        mean_g = Ytest.mean(axis=0)
        index = (sigma_g != 0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = (correlation[index]).mean()

        mid_pred = Mid_Norm.cpu().numpy()
        sigma_p = (mid_pred).std(axis=0)
        mean_p = mid_pred.mean(axis=0)
        correlation_mid = ((mid_pred - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation_mid = (correlation_mid[index]).mean()


        print(
            '|valid_final rse {:5.4f} | valid_final rae {:5.4f} | valid_final corr  {:5.4f}'.format(
                rse, rae, correlation), flush=True)

        print(
            '|valid_mid rse {:5.4f} | valid_mid rae {:5.4f} | valid_mid corr  {:5.4f}'.format(
                rse_mid, rae_mid, correlation_mid), flush=True)
        # if epoch%4==0:
        #     save_model(model, result_file,epoch=epoch)
        return rse, rae, correlation, rse_mid, rae_mid, correlation_mid