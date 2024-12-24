#!/bin/env python
import torch
from torch import nn, _dynamo, utils
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from cythonLoader import cythonLoader as cldr
from torchmetrics.functional import accuracy
#from torchviz import make_dot
def init_weight(m):
    if isinstance(m,nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight) 
class myDataloader(utils.data.Dataset):
    def __init__(self, db_path, row, col, nchs, transform=None):
        self.loader = cldr(db_path)
        self.path = db_path
        self.row = row
        self.col = col
        self.rowl = row
        self.transform = transform
        self.nchs = nchs
    def __getitem__(self, index):
        imgnp, labnp = self.loader.read(index)
        img = torch.tensor(imgnp).reshape([-1, self.row, self.col])[0:self.nchs]
        if self.transform is not None:
            img = self.transform(img)
        return img,labnp[0]
    def __len__(self):
        return self.loader.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'
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
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.Conv_BN_ReLU_2.apply(init_weight)
    def forward(self,x):
        return self.Conv_BN_ReLU_2(x)

class Discreminator(LightningModule):
    def __init__(self, channels=1, level=4):
        super().__init__()
        self.nlevel = level
        out_channels=[channels*4**(i+1) for i in range(self.nlevel+1)]
        maxch = out_channels[-1]
        self.nhead = 10
        self.down = []
        self.conv = []
        #self.loss_func = nn.MSELoss()
        self.loss_func = nn.BCELoss()
        #loss_func = nn.L1Loss()
        self.conv = nn.ModuleList()
        self.down = nn.ModuleList()
        self.conv.append(convLayer(channels, out_channels[0]))
        for x in range(self.nlevel):
            self.down.append(DownsampleLayer(out_channels[x],out_channels[x+1]))
            self.conv.append(convLayer(out_channels[x+1],out_channels[x+1]))
        #self.a = torch.nn.MultiheadAttention(maxch,1)
        self.head = nn.ModuleList()
        for x in range(self.nhead):
            self.head.append(nn.Sequential(
                nn.Linear(maxch, int(maxch)),
                nn.LayerNorm(int(maxch))
            ))
        self.output=nn.Sequential(
            nn.Linear(self.nhead, 1),
            nn.Sigmoid()
        )
        self.output.apply(init_weight)
    def forward(self,input):
        nbatch = input.shape[0]
        nch = input.shape[1]
        chs = torch.Tensor().cuda()
        for ich in range(nch):
            out = input[:,[ich],:,:]
            for x in range(self.nlevel):
                out = self.conv[x](out)
                out = self.down[x](out)
            chs = torch.concatenate((chs,out.flatten()))
        chs = chs.view([nch,nbatch,-1]).transpose(1,0)
        output = torch.Tensor().cuda()
        for h in range(self.nhead):
            head = self.head[h](chs)
            prod = head @ head.transpose(1,2)
            correlation = prod.sum(dim=(1,2))/(torch.vmap(torch.trace)(prod)*(nch-1))-1
            output = torch.concatenate((output,correlation))
        print(output)
        output = self.output(output.view(self.nhead, nbatch).transpose(0,1)/self.nhead)  # -> nbatch, nhead
        return output.flatten()

    def train_dataloader(self):
        trainsz = 31
        bs = 10
        data = myDataloader("./traindb", trainsz,trainsz, 5)
        return DataLoader(data, batch_size = bs, shuffle = False,num_workers = 0,drop_last = True)

    def val_dataloader(self):
        trainsz = 31
        bs = 10
        data = myDataloader("./testdb", trainsz,trainsz, 3)
        return DataLoader(data, batch_size = bs, shuffle = True,num_workers = 0, drop_last = True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 0.01,betas = (0.9, 0.999))
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, cooldown=0, patience=10, min_lr=0.5e-6, eps=1e-8, threshold=1e-4)
        return optimizer
    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "acc", "loss")

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, "val_acc", "val_loss")

    def _shared_eval_step(self, batch, batch_idx, accn, lossn):
        data, target = batch
        output = self.forward(data)
        loss = self.loss_func(output, target)
        output = output.ge(0.5)
        acc = accuracy(output,target,"binary")
        metrics = {accn: acc, lossn: loss}
        self.log_dict(metrics,prog_bar=True)
        return metrics
