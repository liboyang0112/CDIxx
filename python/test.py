#!/usr/bin/env python
import torch
import numpy as np
#from mUNet import mUNet
from UNet import UNet
from unetDataLoader import unetDataLoader as ul
from torch import utils, device, tensor, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL.Image import fromarray
from matplotlib import colors
from torch.nn import functional as F

from PIL import Image
#array = tensor([float(1)]).to(device('cuda:0'))
#net = mUNet(array,channels=1).cuda()
net = UNet(channels=1).cuda()
#data = ul("./traindb", 1, 256,256, 1, 256,256,device('cuda:0'))
ModelSave = 'model_3_2_relu'
net.load_state_dict(torch.load(ModelSave+'/Unet.pt'))
mp = 'turbo'

trainsz = 256
image = Image.open('floatimage.tiff')
x0 = (trainsz-image.width)>>1
y0 = (trainsz-image.height)>>1
#image = image.crop((x0,y0,x1,y1))

image = tensor(np.asarray(image))
image = F.pad(image,(x0,x0,y0,y0),"constant",0)
image = image.view(1,1,trainsz,trainsz).to(device('cuda:0'))
net.eval()
out=net(image)
imgnp=out.cpu()[0][0].detach().numpy()
fromarray(imgnp).save("pattern.tiff")
fig = plt.figure('solved', figsize=[3,3])
subfig = fig.add_subplot(1,1,1)
subfig.imshow(imgnp, norm=colors.LogNorm(), cmap=mp);
fig.savefig("exp.png")

data = ul("./testdb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
dataloader = DataLoader(data, batch_size=4, shuffle=True,num_workers=0,drop_last=True)
saveidx = 2;
for idx in range(0,10):
    img,label=data[idx]
    img=torch.unsqueeze(img,dim=0)
    net.eval()
    datanp=img.cpu()[0][0].detach().numpy()
    out=net(img)
    #plt.gray()
    imgnp=out.cpu()[0][0].detach().numpy()
    labnp=label.cpu()[0].numpy()
    fig = plt.figure('testSample{}'.format(idx), figsize=[11,4])
    subfig = fig.add_subplot(1,3,3)
    subfig.imshow(imgnp, norm=colors.LogNorm(), cmap=mp);
    subfig.set_title("monochromatized pattern");
    subfig = fig.add_subplot(1,3,2)
    subfig.imshow(datanp, norm=colors.LogNorm(), cmap=mp);
    subfig.set_title("broad band pattern");
    subfig = fig.add_subplot(1,3,1)
    subfig.imshow(labnp, norm=colors.LogNorm(), cmap = mp);
    subfig.set_title("monochromatic pattern");
    fig.savefig("trainsample%d.png"%idx)
    #plt.imsave('test.png', np.log2(imgnp)/16+1, vmin=0,vmax=1)
    #plt.imsave('test_data.png', np.log2(datanp)/16+1, vmin=0,vmax=1)
    #plt.imsave('test_lab.png', np.log2(labnp)/16+1,vmin=0,vmax=1)
    if idx == saveidx:
        fromarray(imgnp).save("test.tiff")
        fromarray(datanp).save("test_data.tiff")
        fromarray(labnp).save("test_lab.tiff")

