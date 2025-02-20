#!/bin/env python
import torch
from torch import nn
#from torchviz import make_dot
def init_weight(m):
    if isinstance(m,nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight) 
class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.downsample.apply(init_weight)
    def forward(self,x):
        return self.downsample(x)
class convLayer(nn.Module):
    def __init__(self,in_ch, out_ch):
        super().__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.Conv_BN_ReLU_2.apply(init_weight)
    def forward(self,x):
        return self.Conv_BN_ReLU_2(x)

class Discriminator(nn.Module):
    def __init__(self, channels=2, level=4):
        super().__init__()
        self.nlevel = level
        out_channels=[channels*2*2**(i+1) for i in range(self.nlevel+1)]
        maxch = out_channels[-1]
        self.conv = nn.ModuleList()
        self.down = nn.ModuleList()
        self.conv.append(convLayer(channels, out_channels[0]))
        for x in range(self.nlevel):
            self.down.append(DownsampleLayer(out_channels[x],out_channels[x+1]))
            self.conv.append(convLayer(out_channels[x+1],out_channels[x+1]))
        self.output=nn.Sequential(
            nn.Linear(maxch, 1),
            nn.Sigmoid()
        )
        self.output.apply(init_weight)
    def forward(self,input):
        nbatch = input.shape[0]
        nch = input.shape[1]
        score = torch.zeros(nbatch).cuda()
        for ich in range(nch):
            for jch in range(ich+1, nch):
                out = input[:,[ich,jch],:,:]
                for x in range(self.nlevel):
                    out = self.conv[x](out)
                    out = self.down[x](out)
                score += self.output(out.view(nbatch, -1)).flatten()
        score/=nch*(nch-1)/2
        return score

