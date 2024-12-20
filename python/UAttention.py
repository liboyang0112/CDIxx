#!/bin/env python
from argparse import ArgumentParser
import torch
from torch import device, nn, _dynamo
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from unetDataLoader import unetDataLoader as ul
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
    def forward(self,x):
        return self.Conv_BN_ReLU_2(x)
class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = convLayer(in_ch, in_ch)
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=2),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.upsample(self.conv(x))


class UNet(LightningModule):
    def __init__(self, channels=2, level=4):
        super().__init__()
        self.nlevel = level
        out_channels=[channels*2**(i+2) for i in range(self.nlevel+1)]
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
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels[0],2,kernel_size=3,stride=1,padding=1),
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

    def train_dataloader(self):
        trainsz = 256
        bs = 4
        data = ul("./traindb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
        return DataLoader(data, batch_size = bs, shuffle = False,num_workers = 0,drop_last = True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 0.01,betas = (0.9, 0.999))
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, cooldown=0, patience=10, min_lr=0.5e-6, eps=1e-8, threshold=1e-4)
        return optimizer
    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        #loss_func = nn.BCELoss()
        #loss_func = nn.L1Loss()
        loss_func = nn.MSELoss()
        loss = loss_func(output, target)
        return {'loss': loss}

def main(hparams):
    model = UNet(1,4).cuda()
    trainer = Trainer()
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    main(args)
