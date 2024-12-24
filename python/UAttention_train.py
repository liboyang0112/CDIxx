#!/bin/env python
import torch
from torch import nn,_dynamo,set_float32_matmul_precision
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import Trainer, callbacks,LightningModule
from cythonLoader import cythonLoader as cldr
from UAttention import UAttention
_dynamo.config.suppress_errors = True
set_float32_matmul_precision('high')

class myDataloader(Dataset):
    def __init__(self, db_path, row, col, nchs, transform=None):
        self.loader = cldr(db_path)
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
        return img,img
    def __len__(self):
        return self.loader.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'
class UAttention_l(LightningModule):
    def __init__(self, channels=2, level=3, imgsz = 43):
        super().__init__()
        self.imgsz = imgsz
        #loss_func = nn.BCELoss()
        #loss_func = nn.L1Loss()
        self.loss_func = nn.MSELoss()
        self.nnmodel = UAttention(channels, level)
    def forward(self,x):
        return self.nnmodel(x)
    def train_dataloader(self):
        bs = 1024
        data = myDataloader("./traindb", self.imgsz,self.imgsz, 5)
        return DataLoader(data, batch_size = bs, shuffle = True,num_workers = 15,drop_last = True)

    def val_dataloader(self):
        bs = 100
        data = myDataloader("./testdb", self.imgsz,self.imgsz, 3)
        return DataLoader(data, batch_size = bs, shuffle = False,num_workers = 15, drop_last = True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr = 0.01,betas = (0.9, 0.999))
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.loss_func(output, target)
        metrics = {'loss': loss}
        self.log_dict(metrics,prog_bar=True)
        return metrics
def main():
    model = UAttention_l(2,3,43)
    trainer = Trainer(max_epochs=30, callbacks=[callbacks.ModelCheckpoint(save_top_k=3, monitor='g_loss',enable_version_counter=False, every_n_epochs=10)])
    trainer.fit(model)

if __name__ == '__main__':
    main()
