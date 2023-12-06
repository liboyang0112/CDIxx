#https://blog.csdn.net/kobayashi_/article/details/108951993

from torch import nn
import torch
from torch.nn import functional as F
class mConv2d(nn.Module):
    def __init__(self,array,in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1):
        super(mConv2d, self).__init__()
        self.array = array
        self.in_channels = in_channels;
        self.out_channels = out_channels;
        self.kernel_size = kernel_size
        self.stride = stride;
        self.padding = padding;
        super(mConv2d, self).__init__()
        length = array.size(0);
        self.linear_1 = nn.Sequential(
            nn.Linear(length, length),
            nn.LeakyReLU(inplace=True),
            nn.Linear(length, kernel_size*kernel_size*in_channels*out_channels),
        )
        self.blinear_1 = nn.Sequential(
            nn.Linear(length, length),
            nn.LeakyReLU(inplace=True),
            nn.Linear(length, out_channels),
        )
    def forward(self,x):
        kernel = self.linear_1(self.array).view(self.out_channels,self.in_channels, self.kernel_size, self.kernel_size)
        bias = self.blinear_1(self.array)
        out=F.conv2d(x, kernel, bias, self.stride, self.padding)
        return out


class DownsampleLayer(nn.Module):
    def __init__(self,array, in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        k_size = 3;
        self.Conv_BN_ReLU_2=nn.Sequential(
            mConv2d(array, in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            mConv2d(array, in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            mConv2d(array, in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2
class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out
class mUNet(nn.Module):
    def __init__(self, array, channels=1):
        super(mUNet, self).__init__()
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(array, 1,out_channels[0])#3-64
        self.d2=DownsampleLayer(array, out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(array, out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(array, out_channels[2],out_channels[3])#256-512
        #上采样
        #self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[2],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],1,3,1,1),
            nn.Sigmoid(),
            # BCELoss
        )
    def forward(self,x):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        #out_4,out4=self.d4(out3)
        #out5=self.u1(out4,out_4)
        #out6=self.u2(out5,out_3)
        out6=self.u2(out3,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        return out

