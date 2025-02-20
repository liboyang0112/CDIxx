#!/bin/env python
from torch import set_float32_matmul_precision,utils,tensor, optim, nn
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from cythonLoader import cythonLoader as cldr
from Discriminator import Discriminator
set_float32_matmul_precision('high')
class myDataloader(utils.data.Dataset):
    def __init__(self, db_path, row, col, nchs, transform=None):
        self.loader = cldr(db_path)
        self.row = row
        self.col = col
        self.rowl = row
        self.transform = transform
        self.nchs = nchs
    def __getitem__(self, index):
        imgnp, labnp = self.loader.read(index)
        img = tensor(imgnp).reshape([-1, self.row, self.col])[0:self.nchs]
        if self.transform is not None:
            img = self.transform(img)
        return img,labnp[0]
    def __len__(self):
        return self.loader.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'

class Discriminator_l(LightningModule):
    def __init__(self, channels=2, level=4, imgsz = 43):
        super().__init__()
        self.imgsz = imgsz
        self.nnmodel = Discriminator(channels, level)
        #self.loss_func = nn.MSELoss()
        self.loss_func = nn.BCELoss()
        #loss_func = nn.L1Loss()
    def forward(self, x):
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
        optimizer = optim.Adam(self.parameters(),lr = 0.01,betas = (0.9, 0.999))
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
def main():
    model = Discriminator_l(2,3)
    trainer = Trainer(max_epochs=30, callbacks=[callbacks.ModelCheckpoint(save_top_k=3, monitor='loss',enable_version_counter=False, every_n_epochs=10)])
    trainer.fit(model)

if __name__ == '__main__':
    main()
