#!/bin/env python
import torch
from torch import set_float32_matmul_precision,Tensor, optim
import numpy as np
from cythonLoader import cythonLoader as cldr
from imageIO import writePNG
from GAN import GAN
set_float32_matmul_precision('high')
def toint8(data):
    return np.uint8(((data[0].transpose(0,2).flatten().view(43,43,3).detach().cpu().numpy()+4)/8)*255)
def tofloat(data):
    return (data[0].detach().cpu().numpy()+4)/8
def main():
    version = 53
    epoch = 2989
    step = (epoch+1)*8
    model = GAN.load_from_checkpoint(checkpoint_path=f"lightning_logs/version_{version}/checkpoints/epoch={epoch}-step={step}.ckpt")
    model.eval()
    dataloader = cldr("testdb")
    data, label = dataloader.read(58)
    data = Tensor(data.reshape([1,-1,43,43])[:,0:3,:,:]).cuda()
    cache = np.zeros((data.shape[2],data.shape[3],3), np.int8)
    print(model.discriminator(data), label)
    writePNG("out_orig.png", toint8(data))
    fdata = tofloat(data)
    dataorig = data.clone().detach()
    dataorig.requires_grad_(False)
    for i in range(3):
        writePNG(f"out{i}_orig.png", fdata[i],cache, 0)
    data = model(data)
    fdata = tofloat(data)
    print(model.discriminator(data))
    for i in range(3):
        writePNG(f"out{i}.png", fdata[i], cache, 0, 0)
    writePNG("out.png", toint8(data))

if __name__ == '__main__':
    main()
