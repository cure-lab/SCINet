
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np

class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self, args, in_planes, splitting=True, dropout=0.5,
                 simple_lifting=False):
        super(Interactor, self).__init__()
        self.modified = args.INN
        self.kernel_size = args.kernel
        # dilation = args.dilation
        self.dilation = 1
        self.dropout = args.dropout
        pad = self.dilation * (self.kernel_size - 1) // 2 + 1 # we fix the kernel size of the second layer as 3.
        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        size_hidden = args.hidden_size
        modules_P += [
            nn.ReplicationPad1d(pad),

            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= args.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= args.groups),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad1d(pad),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= args.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= args.groups),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d(pad),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= args.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= args.groups),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad1d(pad),
            nn.Conv1d(in_planes * prev_size, int(in_planes * size_hidden),
                      kernel_size=self.kernel_size, dilation=self.dilation, stride=1, groups= args.groups),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(in_planes * size_hidden), in_planes,
                      kernel_size=3, stride=1, groups= args.groups),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))

            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)

            return (x_even_update, x_odd_update)

        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)

            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)

            return (c, d)


class InteractorLevel(nn.Module):
    def __init__(self, args, in_planes, simple_lifting=False):
        super(InteractorLevel, self).__init__()
        self.level = Interactor(args, in_planes=in_planes, simple_lifting=simple_lifting)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.level(x)

        return (x_even_update, x_odd_update)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, disable_conv):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.disable_conv = disable_conv  # in_planes == out_planes
        if not self.disable_conv:
            self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1,
                                   padding=0, bias=False)

    def forward(self, x):
        if self.disable_conv:
            return self.relu(self.bn1(x))
        else:
            return self.conv1(self.relu(self.bn1(x)))


# class LevelSCINet(nn.Module):
#     def __init__(self, args, in_planes, lifting_size, kernel_size, no_bottleneck,
#                  share_weights, simple_lifting, regu_details, regu_approx):
#         super(LevelSCINet, self).__init__()
#         # self.regu_details = regu_details
#         # self.regu_approx = regu_approx
#         # if self.regu_approx + self.regu_details > 0.0:
#         #     self.loss_details = nn.SmoothL1Loss()

#         self.interact = InteractorLevel(args, in_planes,
#                                           simple_lifting=simple_lifting)
#         self.share_weights = share_weights
#         if no_bottleneck:
#             self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=True)
#         else: # go through another conv. layer
#             self.bootleneck = BottleneckBlock(in_planes, in_planes, disable_conv=False)

#     def forward(self, x):
#         (x_even_update, x_odd_update) = self.interact(x)

#         if self.bootleneck:
#             return self.bootleneck(x_even_update).permute(0, 2, 1), x_odd_update
#         else:
#             return x_even_update.permute(0, 2, 1),x_odd_update

class LevelSCINet(nn.Module):
    def __init__(self, args, in_planes, kernel_size, no_bottleneck, simple_lifting):
        super(LevelSCINet, self).__init__()
        # self.regu_details = regu_details
        # self.regu_approx = regu_approx
        # if self.regu_approx + self.regu_details > 0.0:
        #     self.loss_details = nn.SmoothL1Loss()

        self.interact = InteractorLevel(args, in_planes,
                                        simple_lifting=simple_lifting)
        self.no_bottleneck = no_bottleneck
        # print('bbb',self.no_bottleneck)
        if not no_bottleneck:
            # We still want to do a BN and RELU, but we will not perform a conv
            # as the input_plane and output_plare are the same
            self.bottleneck_even = BottleneckBlock(in_planes, in_planes, disable_conv=True)
            self.bottleneck_odd = BottleneckBlock(in_planes, in_planes, disable_conv=True)
        # else:
        #     self.bottleneck_even = BottleneckBlock(in_planes, in_planes, disable_conv=False)
        #     self.bottleneck_odd = BottleneckBlock(in_planes, in_planes, disable_conv=False)

    def forward(self, x):
        (x_even_update, x_odd_update) = self.interact(x)  # 10 9 128

        if self.no_bottleneck:
            print(x_even_update.shape, x_odd_update.shape)
            return x_even_update.permute(0, 2, 1), x_odd_update.permute(0, 2, 1) #even: B, L, D odd: B, L, D
        else:
            # return self.bottleneck_even(x_even_update).permute(0, 2, 1), x_odd_update.permute(0,2,1)
            return self.bottleneck_even(x_even_update).permute(0, 2, 1), self.bottleneck_odd(x_odd_update.permute(0,2,1))

class SCINet_Tree(nn.Module):
    def __init__(self, args, in_planes, current_layer):
        super().__init__()
        self.current_layer = current_layer
        self.workingblock = LevelSCINet(args=args, in_planes=in_planes,
                    kernel_size=args.kernel, no_bottleneck=True, simple_lifting=False)
        if current_layer!=0:
            self.SCINet_Tree_odd=SCINet_Tree(args, in_planes, current_layer-1)
            self.SCINet_Tree_even=SCINet_Tree(args, in_planes, current_layer-1)
    
    def zip_up_the_pants(self, even, odd):
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2) #L, B, D
        #print(odd.shape)
        even_len = even.shape[0]
        odd_len = odd.shape[0]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[i].unsqueeze(0))
            _.append(odd[i].unsqueeze(0))
        if odd_len < even_len: 
            _.append(even[-1].unsqueeze(0))
        #print(torch.cat(_,0).shape)
        return torch.cat(_,0).permute(1,0,2) #B, L, D
        
    def forward(self, x):
        x_even_update, x_odd_update= self.workingblock(x)
        # We recursively reordered these sub-series. You can run the ./utils/recursive_demo.py to emulate this procedure. 
        if self.current_layer ==0:
            return self.zip_up_the_pants(x_even_update, x_odd_update)
        else:
            return self.zip_up_the_pants(self.SCINet_Tree_even(x_even_update), self.SCINet_Tree_odd(x_odd_update))
        # This part is remastered by Xu Zhijian (zjxu21@cse.cuhk.hk). We have carefully reviewed the code to ensure this part behaves the same as the original one. And we have performed tests on all the 11 datasets and it shows the similar performance. 


class EncoderTree(nn.Module):
    def __init__(self, args, in_planes, num_layers=3, Encoder=True, norm_layer=None):
        super().__init__()
        self.norm = norm_layer
        self.layers=num_layers
        print('layer number:', self.layers)
        # self.count_levels = 0
        # self.ecoder = Encoder
        self.SCINet_Tree = SCINet_Tree(args,in_planes, num_layers-1)

    def forward(self, x):
        x= self.SCINet_Tree(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

# class Chomp1d(nn.Module):
#     def __init__(self, chomp_size):
#         super(Chomp1d, self).__init__()
#         self.chomp_size = chomp_size

#     def forward(self, x):
#         return x[:, :, :-self.chomp_size].contiguous()


class SCINet(nn.Module):
    def __init__(self, args, output_len, input_len, input_dim = 9,
                num_layers = 3, concat_len = 0, no_bootleneck = True):
        super(SCINet, self).__init__()

        self.horizon = output_len
        in_planes = input_dim
        #out_planes = input_dim * (number_levels + 1)
        self.pe = args.positionalEcoding
        self.blocks1 = EncoderTree(args=args, in_planes=in_planes, num_layers=num_layers, Encoder=True)
        if args.stacks == 2: # we only implement two stacks at most.
            self.blocks2 = EncoderTree(args=args, in_planes=in_planes, num_layers=num_layers, Encoder=False)
        self.stacks = args.stacks
        self.concat_len = concat_len

        if no_bootleneck:
            in_planes *= 1

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                # if m.bias is not None:
                m.bias.data.zero_()

        if args.single_step_output_One: # only output the N_th timestep.
            self.projection1 = nn.Conv1d(input_len, 1, groups = args.groups,
                                    kernel_size=1, stride=1, bias=False)
            if args.stacks == 2:
                if self.concat_len:
                    self.projection2 = nn.Conv1d(concat_len + output_len, 1, groups = args.groups,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(input_len + output_len, 1, groups = args.groups,
                                                kernel_size = 1, bias = False)
        else: # output the N timesteps.
            self.projection1 = nn.Conv1d(input_len, output_len, groups = args.groups,
                                    kernel_size=1, stride=1, bias=False)
            if args.stacks == 2:                   
                if self.concat_len:
                    self.projection2 = nn.Conv1d(concat_len + output_len, output_len, groups = args.groups,
                                                kernel_size = 1, bias = False)
                else:
                    self.projection2 = nn.Conv1d(input_len + output_len, output_len, groups = args.groups,
                                                kernel_size = 1, bias = False)

        # For positional encoding
        self.hidden_size = in_planes
        if self.hidden_size % 2 == 1:
            self.hidden_size += 1
    
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                max(num_timescales - 1, 1))
        temp = torch.arange(num_timescales, dtype=torch.float32)
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)
    
    def get_position_encoding(self, x):
        max_length = x.size()[1]
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
        temp1 = position.unsqueeze(1)  # 5 1
        temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)  #[T, C]
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
    
        return signal

    def forward(self, x):
        if self.pe:
            pe = self.get_position_encoding(x)
            if pe.shape[2] > x.shape[2]:
                x += pe[:, :, :-1]
            else:
                x += self.get_position_encoding(x)

        # the first stack
        res1 = x
        x = self.blocks1(x)
        x += res1
        x = self.projection1(x)

        if self.stacks == 1:
            return x

        elif self.stacks == 2:
            MidOutPut = x
            if self.concat_len:
                x = torch.cat((res1[:, -self.concat_len:,:], x), dim=1)
            else:
                x = torch.cat((res1, x), dim=1)

            # the second stack
            res2 = x
            x = self.blocks2(x)
            x += res2
            x = self.projection2(x)
            return x, MidOutPut


# class SCINetEncoder(nn.Module):
#     def __init__(self, args, output_len, input_len, input_dim=9,
#                 num_layers = 3,
#                 concat_len = None, no_bootleneck=True):
#         super(SCINetEncoder, self).__init__()
#         in_planes = input_dim
#         #out_planes = input_dim * (number_levels + 1)
#         self.pe = args.positionalEcoding
#         self.blocks1 = EncoderTree(args=args, in_planes=in_planes, num_layers=num_layers, Encoder=True)

#         self.concat_len = concat_len

#         if no_bootleneck:
#             in_planes *= 1

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 # nn.init.xavier_uniform_(m.weight.data)
#                 # if m.bias is not None:
#                 m.bias.data.zero_()
#         if args.single_step_output_One:
#             self.projection1 = nn.Conv1d(input_len, 1,
#                                     kernel_size=1, stride=1, bias=False)
#         else:
#             self.projection1 = nn.Conv1d(input_len, output_len,
#                                     kernel_size=1, stride=1, bias=False)

#         self.hidden_size = in_planes
#         # For positional encoding
#         if self.hidden_size % 2 == 1:
#             self.hidden_size += 1
    
#         num_timescales = self.hidden_size // 2 
#         max_timescale = 10000.0
#         min_timescale = 1.0

#         log_timescale_increment = (
#                 math.log(float(max_timescale) / float(min_timescale)) /
#                 max(num_timescales - 1, 1)) 
#         temp = torch.arange(num_timescales, dtype=torch.float32)
#         inv_timescales = min_timescale * torch.exp(
#             torch.arange(num_timescales, dtype=torch.float32) *
#             -log_timescale_increment)
#         self.register_buffer('inv_timescales', inv_timescales)
    
#     def get_position_encoding(self, x):
#         max_length = x.size()[1]
#         position = torch.arange(max_length, dtype=torch.float32,
#                                 device=x.device)  # tensor([0., 1., 2., 3., 4.], device='cuda:0')
#         temp1 = position.unsqueeze(1)  # 5 1
#         temp2 = self.inv_timescales.unsqueeze(0)  # 1 256
#         scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # 5 256
#         signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
#                            dim=1)  # 5 512 [T, C]
#         signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
#         signal = signal.view(1, max_length, self.hidden_size)

#         return signal

    # def forward(self, x):
    #     if self.pe:
    #         pe = self.get_position_encoding(x)
    #         if pe.shape[2] > x.shape[2]:
    #             x += pe[:, :, :-1]
    #         else:
    #             x += self.get_position_encoding(x)

    #     res1 = x

    #     x = self.blocks1(x)

    #     hid = x
    #     x += res1

    #     x = self.projection1(x)

    #     return x

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=12)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--groups', type=int, default=1)

    parser.add_argument('--hidden-size', default=1, type=int, help='hidden channel of module')
    parser.add_argument('--INN', default=1, type=int, help='use INN or basic strategy')
    parser.add_argument('--kernel', default=3, type=int, help='kernel size')
    parser.add_argument('--dilation', default=1, type=int, help='dilation')
    parser.add_argument('--positionalEcoding', type=bool, default=True)

    parser.add_argument('--single_step_output_One', type=int, default=0)

    args = parser.parse_args()

    model = SCINet(args, output_len=24, input_len= 96, input_dim=8,
                   num_layers=3, concat_len=None, no_bottleneck=True).cuda()
    x = torch.randn(32, 96, 8).cuda()
    y = model(x)
    print(y.shape)
