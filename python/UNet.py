from torch import nn
import torch
class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.downsample(x)
class convLayer(nn.Module):
    def __init__(self,in_ch, out_ch):
        super().__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.Conv_BN_ReLU_2(x)
class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.upsample(self.Conv_BN_ReLU_2(x))

class UNet(nn.Module):
    def __init__(self, channels=1, level=7):
        super().__init__()
        self.nlevel = level
        out_channels=[channels*2**(i+6) for i in range(self.nlevel+1)]
        self.d = []
        self.u = []
        self.c = []
        c = []
        d = []
        u = []
        c.append(convLayer(channels, out_channels[0]).cuda())
        for x in range(self.nlevel):
            d.append(DownsampleLayer(out_channels[x],out_channels[x+1]).cuda())
            c.append(convLayer(out_channels[x+1],out_channels[x+1]).cuda())
        self.d = nn.ModuleList(d)
        self.c = nn.ModuleList(c)
        u.append(UpSampleLayer(out_channels[self.nlevel],out_channels[self.nlevel-1]).cuda())
        for x in range(self.nlevel-1):
            u.append(UpSampleLayer(out_channels[self.nlevel-x],out_channels[self.nlevel-x-2]).cuda())
        self.u = nn.ModuleList(u)
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels[0],1,3,1,1),
            nn.Sigmoid(),
        )
    def forward(self,out):
        ds = []
        for x in range(self.nlevel):
            out = self.c[x](out)
            ds.append(out)
            out = self.d[x](out)
        for x in range(self.nlevel):
            out = torch.cat((self.u[x](out),ds[self.nlevel - x - 1]), dim=1)
            del ds[self.nlevel - x - 1]
        return self.o(out)

