# Initial comment
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

class NormType(Enum):
    BATCHNORM = 1
    LAYERNORM = 2
    GROUPNORM = 3

def get_norm_layer(norm_type: NormType, out_channels : int, ln_shape = None):
    norm_layer = None
    if norm_type == NormType.BATCHNORM:
        norm_layer = nn.BatchNorm2d(num_features=out_channels, affine=False)
    elif norm_type == NormType.LAYERNORM:
        norm_layer = nn.LayerNorm(normalized_shape=ln_shape, elementwise_affine=False)
    elif norm_type == NormType.GROUPNORM:
        num_groups = 2
        norm_layer = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=False)

    return norm_layer

dropout_value = 0.1
class Net(nn.Module):
    @staticmethod
    def calc_activation_shape(inp_dim, ksize, dilation=(1, 1), stride=(1, 1), padding=(0, 0)):
        def shape_each_dim(i):
            odim_i = inp_dim[i] + 2 * padding[i] - dilation[i] * (ksize[i] - 1) - 1
            return int(odim_i / stride[i]) + 1

        return shape_each_dim(0), shape_each_dim(1)

    def __init__(self, norm_type : NormType):
        super(Net, self).__init__()

        # Input Block
        in_channels=1
        inp_dim = (28,28)
        out_channels=8
        kernel_size=(3, 3)
        conv_output_shape = Net.calc_activation_shape(inp_dim=inp_dim, ksize=kernel_size)
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_type=norm_type, out_channels=out_channels, ln_shape=[out_channels, *conv_output_shape]),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        in_channels = out_channels
        inp_dim = conv_output_shape
        out_channels=32
        kernel_size=(3, 3)
        conv_output_shape = Net.calc_activation_shape(inp_dim=inp_dim, ksize=kernel_size)
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.ReLU(),
            get_norm_layer(norm_type=norm_type, out_channels=out_channels, ln_shape=[out_channels, *conv_output_shape]),
            nn.Dropout(dropout_value)
        ) # output_size = 24
        print(self.convblock2)

        # TRANSITION BLOCK 1
        in_channels = out_channels
        inp_dim = conv_output_shape
        out_channels=10
        kernel_size=(1, 1)
        conv_output_shape = Net.calc_activation_shape(inp_dim=inp_dim, ksize=kernel_size)
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        in_channels = out_channels
        inp_dim = conv_output_shape
        out_channels=16
        kernel_size=(3, 3)
        conv_output_shape = Net.calc_activation_shape(inp_dim=inp_dim, ksize=kernel_size)
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.ReLU(),            
            get_norm_layer(norm_type=norm_type, out_channels=out_channels, ln_shape=[out_channels, *conv_output_shape]),
            nn.Dropout(dropout_value)
        ) # output_size = 10

        in_channels = out_channels
        inp_dim = conv_output_shape
        out_channels=16
        kernel_size=(3, 3)
        conv_output_shape = Net.calc_activation_shape(inp_dim=inp_dim, ksize=kernel_size)
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.ReLU(),            
            get_norm_layer(norm_type=norm_type, out_channels=out_channels, ln_shape=[out_channels, *conv_output_shape]),
            nn.Dropout(dropout_value)
        ) # output_size = 8

        in_channels = out_channels
        inp_dim = conv_output_shape
        out_channels=16
        kernel_size=(3, 3)
        conv_output_shape = Net.calc_activation_shape(inp_dim=inp_dim, ksize=kernel_size)
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.ReLU(),            
            get_norm_layer(norm_type=norm_type, out_channels=out_channels, ln_shape=[out_channels, *conv_output_shape]),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        # self.convblock7 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
        #     nn.ReLU(),            
        #     nn.BatchNorm2d(16),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 6
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=conv_output_shape[0])
        ) # output_size = 1

        in_channels = out_channels
        out_channels=10
        kernel_size=(1, 1)
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=0, bias=False)
        ) 

        # self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        # x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

