import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
from torch.nn.utils import weight_norm
import argparse
import numpy as np

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        # self.conv_even = lambda x: x[:, ::2, :]
        # self.conv_odd = lambda x: x[:, 1::2, :]
        # To simplify, we removed the dimensions except for the length dimension. 

    def even(self, x):
        return x[::2]

    def odd(self, x):
        return x[1::2]

    def forward(self, x):
        '''Returns the odd and even part'''
        return self.even(x), self.odd(x)

class SCINet_Tree(nn.Module):
    def __init__(self, args, in_planes, current_layer):
        super().__init__()
        self.current_layer=current_layer
        self.workingblock=Splitting() # To simplyfy, we replaced the actual SCINetblock with a splitting block. 
        if current_layer!=0:
            self.SCINet_Tree_odd=SCINet_Tree(args, in_planes, current_layer-1)
            self.SCINet_Tree_even=SCINet_Tree(args, in_planes, current_layer-1)
    
    def zip_up_the_pants(self, even, odd):
        even_len=even.shape[0]
        odd_len=odd.shape[0]
        mlen=min((odd_len,even_len))
        _=[]
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len<even_len: 
            _.append(even[-1].unsqueeze(0))
        #print(torch.cat(_,0).shape)
        return torch.cat(_,0)
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        print(self.current_layer,'layer splitting: ',x_even_update,x_odd_update)
        if self.current_layer ==0:
            a=self.zip_up_the_pants(x_even_update, x_odd_update)
            print(self.current_layer,'layer joining',a)
            return a
        else:
            a=self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))
            print(self.current_layer,'layer joining',a)
            return a

if __name__ == '__main__':
    args=None
    in_planes=None
    model=SCINet_Tree(args, in_planes, 2)
    x=[12,435,765,2323,45,234,456,567] #input your random time series
    x=torch.tensor(x)
    print(model(x))
