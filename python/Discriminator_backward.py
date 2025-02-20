#!/bin/env python
import torch
from torch import set_float32_matmul_precision,Tensor, optim
import numpy as np
from cythonLoader import cythonLoader as cldr
from imageIO import writePNG
from Discreminator import Discreminator
set_float32_matmul_precision('high')
def toint8(data):
    return np.uint8(((data[0].transpose(0,2).flatten().view(43,43,3).detach().cpu().numpy()+4)/8)*255)
def tofloat(data):
    return (data[0].detach().cpu().numpy()+4)/8
def main():
    model = Discreminator.load_from_checkpoint(checkpoint_path="lightning_logs/version_1/checkpoints/epoch=39-step=160.ckpt")
    model.eval()
    dataloader = cldr("testdb")
    data, label = dataloader.read(24)
    data = Tensor(data.reshape([1,-1,43,43])[:,0:3,:,:]).cuda()
    data.requires_grad=True
    cache = np.zeros((data.shape[2],data.shape[3],3), np.int8)
    print(model(data), label)
    writePNG("out_orig.png", toint8(data))
    fdata = tofloat(data)
    datadiff = data.detach().cpu().numpy()
    dataorig = data.clone().detach()
    dataorig.requires_grad_(False)

    data0 = data[:,[0],:,:].detach()
    data0.requires_grad = False
    data12 = data[:,[1,2],:,:].detach()
    data12.requires_grad = True
    myoptimizer = optim.Adam([data12],lr = 0.01,betas = (0.9, 0.999))
    for i in range(3):
        writePNG(f"out{i}_orig.png", fdata[i],cache, 0)
    for iter in range(100):
        data01 = torch.concatenate([data12[:,[0],:,:], data0], 1)
        data02 = torch.concatenate([data12[:,[1],:,:], data0], 1)
        y = 2-model(data01)-model(data02)
        print(y)
        #diff = ((data-dataorig)**2).sum()
        #print(y,diff)
        #y+=diff
        y.backward()
        myoptimizer.step()
    data = torch.concatenate([data0, data12], 1)
    datadiff = data.detach().cpu().numpy() - datadiff
    fdata = tofloat(data)
    print(model(data), label)
    for i in range(3):
        writePNG(f"out{i}.png", fdata[i], cache, 0, 0)
    writePNG("out.png", toint8(data))

if __name__ == '__main__':
    main()
