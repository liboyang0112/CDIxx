#!/bin/env python
from argparse import ArgumentParser
import torch
from torch import nn, _dynamo, utils
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, callbacks
from cythonLoader import cythonLoader as cldr
#from torchviz import make_dot
_dynamo.config.suppress_errors = True
def init_weight(m):
    if isinstance(m,nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight) 
class myDataloader(utils.data.Dataset):
    def __init__(self, db_path, row, col, device, transform=None):
        self.loader = cldr(db_path)
        self.path = db_path
        self.row = row
        self.col = col
        self.rowl = row
        self.device = device
        self.transform = transform
    def __getitem__(self, index):
        imgnp, labnp = self.loader.read(index)
        img = torch.tensor(imgnp).to(self.device).reshape([-1, self.row, self.col])[0:3]
        if self.transform is not None:
            img = self.transform(img)
        return img,labnp[0]
    def __len__(self):
        return self.loader.length
        #return 10
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

class ImgAttention(LightningModule):
    def __init__(self, channels=1, level=4):
        super().__init__()
        self.nlevel = level
        out_channels=[channels*4**(i+1) for i in range(self.nlevel+1)]
        maxch = out_channels[-1]
        self.down = []
        self.conv = []
        self.loss_func = nn.MSELoss()
        #self.loss_func = nn.BCELoss()
        #loss_func = nn.L1Loss()
        self.conv = nn.ModuleList()
        self.down = nn.ModuleList()
        self.conv.append(convLayer(channels, out_channels[0]).cuda())
        for x in range(self.nlevel):
            self.down.append(DownsampleLayer(out_channels[x],out_channels[x+1]).cuda())
            self.conv.append(convLayer(out_channels[x+1],out_channels[x+1]).cuda())
        #self.a = torch.nn.MultiheadAttention(maxch,1)
        self.transformer = nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(maxch, 1),4)
        self.output=nn.Sequential(
            nn.Linear(maxch, maxch),
            nn.LeakyReLU(inplace=True),
            nn.Linear(maxch, maxch),
            nn.LeakyReLU(inplace=True),
            nn.Linear(maxch, 1),
        )
        self.output.apply(init_weight)
    def forward(self,input):
        nbatch = input.shape[0]
        nch = input.shape[1]
        chs = torch.Tensor().to("cuda:0")
        for ich in range(nch):
            out = input[:,[ich],:,:]
            for x in range(self.nlevel):
                out = self.conv[x](out)
                out = self.down[x](out)
            chs = torch.concatenate((chs,out.flatten()))
        chs = chs.view([nch,nbatch,-1])
        #afterat,_ = self.a(chs,chs,chs)
        afterat = self.transformer(chs)
        afterat = afterat.transpose(1,0)
        afterat = self.output(afterat).view([nbatch,3])
        values = afterat
        return values.mean(dim=1)

    def train_dataloader(self):
        trainsz = 31
        bs = 10
        data = myDataloader("./traindb", trainsz,trainsz, 'cuda:0')
        return DataLoader(data, batch_size = bs, shuffle = False,num_workers = 0,drop_last = True)

    def validation_dataloader(self):
        trainsz = 31
        bs = 4
        data = myDataloader("./testdb", trainsz,trainsz, 'cuda:0')
        return DataLoader(data, batch_size = bs, shuffle = True,num_workers = 0,drop_last = True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 0.01,betas = (0.9, 0.999))
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, cooldown=0, patience=10, min_lr=0.5e-6, eps=1e-8, threshold=1e-4)
        return optimizer
    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = {'loss':self.loss_func(output, target)}
        self.log('loss', loss['loss'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.loss_func(output, target)
        pred = torch.argmax(output, dim=1)
        self.val_acc.update(pred,target)
        self.log("test_loss", loss,prog_bar=True)
        self.log("test_acc", self.val_acc,prog_bar=True)
        return {'loss': loss}

def main(hparams):
    model = ImgAttention(1,4).cuda()
    #dummy_input = torch.randn(4,3,31,31).to('cuda:0')
    #dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    #dot.format = 'png'
    #dot.render('simple_net')
    trainer = Trainer(callbacks=[callbacks.ModelCheckpoint(save_top_k=0)])
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=None)
    args = parser.parse_args()

    main(args)
