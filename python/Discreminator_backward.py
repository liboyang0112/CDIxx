#!/bin/env python
from torch import set_float32_matmul_precision,Tensor, optim
import numpy as np
from cythonLoader import cythonLoader as cldr
from imageIO import writePNG
from Discreminator import Discreminator
set_float32_matmul_precision('high')
def toint8(data):
    return np.uint8(((data[0].transpose(0,2).flatten().view(31,31,3).detach().cpu().numpy()+4)/8)*255)
def tofloat(data):
    return (data[0].detach().cpu().numpy()+4)/8
def main():
    model = Discreminator.load_from_checkpoint(checkpoint_path="lightning_logs/version_4/checkpoints/epoch=9-step=5000.ckpt")
    model.eval()
    dataloader = cldr("testdb")
    data, label = dataloader.read(22)
    data = Tensor(data.reshape([1,-1,31,31])[:,0:3,:,:]).cuda()
    data.requires_grad=True
    cache = np.zeros((data.shape[2],data.shape[3],3), np.int8)
    print(model(data), label)
    myoptimizer = optim.Adam([data],lr = 0.01,betas = (0.9, 0.999))
    writePNG("out_orig.png", toint8(data))
    fdata = tofloat(data)
    datadiff = data.detach().cpu().numpy()
    for i in range(3):
        writePNG(f"out{i}_orig.png", fdata[i],cache, 0)
    for iter in range(4):
        y = 1-model(data)
        print(y)
        y.backward()
        myoptimizer.step()
    datadiff = data.detach().cpu().numpy() - datadiff
    fdata = tofloat(data)
    print(model(data), label)
    for i in range(3):
        writePNG(f"out{i}.png", fdata[i], cache, 0, 0)
    writePNG("out.png", toint8(data))

if __name__ == '__main__':
    main()
