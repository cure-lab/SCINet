
import os
import numpy as np
import torch

def save_model(epoch, lr, model, model_dir, model_name='pems08', horizon=12):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    file_name = os.path.join(model_dir, model_name+str(horizon)+'.bin')
    torch.save(
        {
        'epoch': epoch,
        'lr': lr,
        'model': model.state_dict(),
        }, file_name)
    print('save model in ',file_name)


def load_model(model, model_dir, model_name='pems08', horizon=12):
    if not model_dir:
        return
    file_name = os.path.join(model_dir, model_name+str(horizon)+'.bin') 

    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        print('loaded the model...', file_name, 'now lr:', lr, 'now epoch:', epoch)
    return model, lr, epoch

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj==1:
        lr_adjust = {epoch: args.lr * (0.95 ** (epoch // 1))}

    elif args.lradj==2:
        lr_adjust = {
            0: 0.0001, 5: 0.0005, 10:0.001, 20: 0.0001, 30: 0.00005, 40: 0.00001
            , 70: 0.000001
        }

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
    return lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean
