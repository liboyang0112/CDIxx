#!/usr/bin/env python
import torch
from os.path import exists
from os import mkdir
import numpy as np
#from mUNet import mUNet
from UNet import UNet
from unetDataLoader import unetDataLoader as ul
from torch import device, tensor
from torch.utils.data import DataLoader
from libimageIO_cython import writeFloat,readImage,writePng
from torch.nn import functional as F
from torchvision.transforms import CenterCrop

#array = tensor([float(1)]).to(device('cuda:0'))
#net = mUNet(array,channels = 1).cuda()
net = UNet(1,4).cuda()
#data = ul("./traindb", 1, 256,256, 1, 256,256,device('cuda:0'))
ModelSave = 'model_4_2_linear'
net.load_state_dict(torch.load(ModelSave + '/Unet.pt'))
mp = 'turbo'

trainsz = 256
runExp = 0
if runExp:
    image = readImage('broad_pattern.bin')
    x0 = (trainsz - image.shape[0]) >> 1
    y0 = (trainsz - image.shape[1]) >> 1
    
    image = tensor(np.asarray(image))
    if x0 > 0:
        image = F.pad(image,(x0,x0,y0,y0),"constant",0)
    elif x0 < 0:
        crp = CenterCrop(trainsz)
        image = crp(image)
    
    image = image.view(1,1,trainsz,trainsz).to(device('cuda:0'))
    imgnp = image.cpu()[0][0].numpy()
    writeFloat("cropped.bin",imgnp)
    net.eval()
    out = net(image)
    imgnp = out.cpu()[0][0].detach().numpy()
    writeFloat("pattern.bin",imgnp)

data = ul("./testdb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
dataloader = DataLoader(data, batch_size = 4, shuffle = True,num_workers = 0,drop_last = True)
cache = np.zeros((trainsz, trainsz, 3), np.int8())
for idx in range(0,10):
    img,label = data[idx]
    img = torch.unsqueeze(img,dim = 0)
    net.eval()
    datanp = img.cpu()[0][0].detach().numpy()
    #plt.gray()
    imgnp = net(img).cpu()[0][0].detach().numpy()
    labnp = label.cpu()[0].numpy()

    dirsave = 'test%d'%idx
    if not exists(dirsave):
        mkdir(dirsave)
    writeFloat(dirsave + "/test.bin",imgnp)
    writeFloat(dirsave + "/test_lab.bin",labnp)
    writePng(dirsave + "/test_broad", datanp, cache, 1, 1)
