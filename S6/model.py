import torch.nn as nn
import torch.nn.functional as F

# Depthwise Convolution class
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(DepthwiseSeparable, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=(3, 3), padding=1, stride=self.stride, groups=self.in_ch, bias=False),
            nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch, kernel_size=(1, 1), bias=False)
        )

    def forward(self, x):
        return self.depthwise(x)
    
# The network
class Net(nn.Module):
    def __init__(self, drop=0.025):
        super(Net, self).__init__()

        # Define C1 C2 C3 C4 Output blocks
        self.conblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # I: 32x32x3 | O: 32x32x32 | RF: 3x3
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=drop),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # I: 32x32x32 | O: 32x32x32 | RF: 5x5
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=drop)
        )

        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, bias=False), #I: 32x32x32 | O: 16x16x32 | RF: 7x7
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=drop)
        )
        
        self.conblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), # I: 16x16x32 | O: 16x16x64 | RF: 11x11
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=drop)
        )

        self.transblock2 = nn.Sequential(
            DepthwiseSeparable(in_ch=64, out_ch=64, stride=2), #I: 16x16x64 | O: 8x8x64 | RF: 15x15
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=drop)
        )

        self.conblock3 = nn.Sequential(
            DepthwiseSeparable(in_ch=64, out_ch=128), #I: 8x8x64 | O: 8x8x128 | RF: 23x23
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=drop),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), bias=False), # I: 8x8x128 | O: 8x8x32 | RF: 23x23
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=drop)
        )

        self.transblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # I: 8x8x32 | O: 6x6x32 | RF: 39x39
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, dilation=2, bias=False), # I: 6x6x32 | O: 4x4x32 | RF: 55x55
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=drop)
        )

        self.conblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), # I: 4x4x32 | O: 4x4x32 | RF: 63x63
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(3, 3), padding=1, bias=False), # I: 4x4x32 | O: 4x4x10 | RF: 71x71
            
        )

        self.gap = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        x = self.transblock1(self.conblock1(x))
        x = self.transblock2(self.conblock2(x))
        x = self.transblock3(self.conblock3(x))
        x = self.conblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10)

        return F.log_softmax(input=x, dim=-1)
