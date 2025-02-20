#!/bin/env python
import torch
from torch import nn, _dynamo
_dynamo.config.suppress_errors = True
class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.downsample(x)
class convLayer(nn.Module):
    def __init__(self,in_ch, out_ch, ks = 3):
        super().__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=ks,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=ks, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.Conv_BN_ReLU_2(x)
class TconvLayer(nn.Module):
    def __init__(self,in_ch, out_ch, ks = 3):
        super().__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=ks,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=ks, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.Conv_BN_ReLU_2(x)
class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.upsample(x)


class UAttention(nn.Module):
    def __init__(self, channels=2, level=3):
        super().__init__()
        self.nlevel = level
        out_channels=[channels*4**(i+1) for i in range(self.nlevel+1)]
        self.d = nn.ModuleList()
        self.u = nn.ModuleList()
        self.c = nn.ModuleList()
        self.ct = nn.ModuleList()
        self.c.append(convLayer(channels, out_channels[0]).cuda())
        for x in range(self.nlevel):
            self.d.append(DownsampleLayer(out_channels[x],out_channels[x+1]).cuda())
            self.c.append(convLayer(out_channels[x+1],out_channels[x+1]).cuda())
        for x in range(self.nlevel):
            self.u.append(UpSampleLayer(out_channels[self.nlevel-x],out_channels[self.nlevel-x-1]).cuda())
            self.ct.append(TconvLayer(out_channels[self.nlevel-x-1]*2,out_channels[self.nlevel-x-1]).cuda())
        self.o=nn.Sequential(
            nn.ConvTranspose2d(out_channels[0],out_channels[0],kernel_size=3,stride=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels[0], 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(2),
        )
    def __UNetForward(self,out):
        ds = []
        for x in range(self.nlevel):
            out = self.c[x](out)
            ds.append(out)
            out = self.d[x](out)
        for x in range(self.nlevel):
            out = self.ct[x](torch.cat([self.u[x](out),ds[self.nlevel - x - 1]], dim=1))
            del ds[self.nlevel - x - 1]
        return self.o(out)

    def forward(self,input):
        nch = input.shape[1]
        out = torch.zeros(input.shape).cuda()
        for ich in range(nch):
            for jch in range(ich+1,nch):
                outij = input[:,[ich,jch],:,:]
                outij = self.__UNetForward(outij)
                out[:,[ich,jch],:,:] += outij
        out /= nch
        return input + out

