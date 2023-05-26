#!/usr/bin/env python
import torch
import numpy as np
#from mUNet import mUNet
from UNet import UNet
from libunetDataLoader_cython import unetDataLoader as ul
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
net.load_state_dict(torch.load('SAVE/Unet.pt'))
mp = 'turbo'

image = Image.open('floatimage.tiff')
x0 = (512-image.width)>>1
y0 = (512-image.height)>>1
#image = image.crop((x0,y0,x1,y1))

image = tensor(np.asarray(image))
image = F.pad(image,(x0,x0,y0,y0),"constant",0)
image = image.view(1,1,512,512).to(device('cuda:0'))
net.eval()
out=net(image)
imgnp=out.cpu()[0][0].detach().numpy()
fromarray(imgnp).save("pattern.tiff")
fig = plt.figure('solved', figsize=[3,3])
subfig = fig.add_subplot(1,1,1)
subfig.imshow(imgnp, norm=colors.LogNorm(), cmap=mp);
fig.savefig("exp.png")

