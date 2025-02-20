#!/bin/env python
from pytorch_lightning import LightningModule
from Discriminator import Discriminator
from UAttention import UAttention as Generator
class GAN(LightningModule):
    def __init__(
            self,
            **kwargs,
            ):
        super().__init__()
        # networks
        self.generator = Generator(2,3)
        self.discriminator = Discriminator(2,3)

    def forward(self, z):
        return self.generator(z)

