#!/bin/env python
import torch, torchvision
import pytorch_lightning as pl
from cythonLoader import cythonLoader as cldr
from pytorch_lightning import Trainer, callbacks
from torch.utils.data import DataLoader, Dataset
from GAN import GAN
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
        return img,labnp
    def __len__(self):
        return self.loader.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'
class GAN_pl(pl.LightningModule):
    def __init__(
        self,
        imgsz,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.imgsz = imgsz

        gan = GAN()

        # networks
        self.generator = gan.generator
        self.discriminator = gan.discriminator

        self.regularization_term = torch.nn.MSELoss()
        self.loss_func = torch.nn.BCELoss()

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return self.loss_func(y_hat, y)

    def training_step(self, batch):
        imgs, labels = batch
        labels = labels.flatten()
        seeder = imgs[torch.where(labels==0)[0],:,:,:]
        imgs = imgs[torch.where(labels==1)[0],:,:,:]

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        seeder = seeder.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(seeder)

        # log sampled images
        sample_imgs = self.generated_imgs[:6,0:2,:,:]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("train/generated_images", grid, self.current_epoch)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(seeder.size(0))
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.regularization_term(self.discriminator(self.generated_imgs), valid)
        g_loss += 0.001*self.regularization_term(seeder, self.generated_imgs)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0))
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(seeder.size(0))
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self.generated_imgs.detach()), fake)
        fake_loss += self.adversarial_loss(self.discriminator(seeder.detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 3
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def train_dataloader(self):
        bs = 1024
        data = myDataloader("./traindb", self.imgsz,self.imgsz, 5)
        return DataLoader(data, batch_size = bs, shuffle = True,num_workers = 15,drop_last = True)

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        pass
        #z = self.validation_z.type_as(self.generator.model[0].weight)
        ## log sampled images
        #sample_imgs = self(z)
        #grid = torchvision.utils.make_grid(sample_imgs)
        #self.logger.experiment.add_image("validation/generated_images", grid, self.current_epoch)

def main():
    version = 53
    epoch = 2989
    step = (epoch+1)*8
    model = GAN_pl.load_from_checkpoint(checkpoint_path=f"lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt")
    trainer = Trainer(max_epochs=3000, callbacks=[callbacks.ModelCheckpoint(save_top_k=-1, monitor='d_loss',enable_version_counter=False, every_n_epochs=10)])
    trainer.fit(model)

if __name__ == '__main__':
    main()
