#!/usr/bin/env python
import torch
from os.path import exists
from os import mkdir
#from mUNet import mUNet
from UNet import UNet
from torch import utils, device, tensor, nn
from unetDataLoader import unetDataLoader as ul
from torch.utils.data import DataLoader
import torchvision.transforms as T
#from torch.utils.tensorboard import SummaryWriter
from matplotlib import colors
import matplotlib.pyplot as plt
from PIL import Image
from PIL.Image import fromarray
import numpy as np

#writer = SummaryWriter()
#array = tensor([float(1)]).to(device('cuda:0'))
#net = mUNet(array,channels=1).cuda()
net = UNet(channels=1).cuda()
optimizer = torch.optim.Adam(net.parameters(),lr=0.0001,betas=(0.9, 0.999))
loss_func = nn.BCELoss()
#loss_func = nn.L1Loss()
#loss_func = nn.MSELoss()
trainsz = 256
data = ul("./traindb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
datatest = ul("./testdb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
dataloader = DataLoader(data, batch_size=4, shuffle=True,num_workers=0,drop_last=True)
testloader = DataLoader(datatest, batch_size=4, shuffle=True,num_workers=0,drop_last=True)
EPOCH=1000
print('load net')
testIdx = 1;
ModelSave = 'model_4_2'
if not exists(ModelSave):
    mkdir(ModelSave)
if not exists('Log_imgs'):
    mkdir('Log_imgs')
if exists(ModelSave+'/Unet.pt'):
    net.load_state_dict(torch.load(ModelSave+'/Unet.pt'))
print('load success')
imgval,label=datatest[testIdx]
mp = 'turbo'
fig = plt.figure('label', figsize=[3,3])
subfig = fig.add_subplot(1,1,1)
subfig.imshow(label.cpu()[0].numpy(), norm=colors.LogNorm(), cmap=mp);
fig.savefig("Log_imgs/seglab.png")
imgtest=torch.unsqueeze(imgval,dim=0)
testimg, testlabel = next(iter(testloader))
train_losses = []
test_losses = []

image = Image.open('floatimage.tiff')
x0 = (trainsz-image.width)>>1
y0 = (trainsz-image.height)>>1
#image = image.crop((x0,y0,x1,y1))

transform = T.Resize(trainsz)
image = tensor(np.asarray(transform(image)))
image = image.view(1,1,trainsz,trainsz).to(device('cuda:0'))

imagesim = Image.open('simfloat.tiff')
x0 = (trainsz-imagesim.width)>>1
y0 = (trainsz-imagesim.height)>>1

imagesim = tensor(np.asarray(imagesim))
imagesim = torch.nn.functional.pad(imagesim,(x0,x0,y0,y0),"constant",0)
imagesim = imagesim.view(1,1,trainsz,trainsz).to(device('cuda:0'))
for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    net.train()
    total_loss = 0;
    for i,(img,label) in  enumerate(dataloader):
        loss=loss_func(net(img),label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #train_losses.append(loss.item())
        if i%20==0:
            print("training loss = %f"%(loss.item()*1e6))
        total_loss += loss.item()*2e3
        loss=loss_func(net(testimg), testlabel)
        #test_losses.append(loss.item())
        if i%20==0:
            print("validation loss = %f"%(loss.item()*1e6))
    print("total training loss = %f"%total_loss)
    torch.save(net.state_dict(),ModelSave+'/Unet.pt')
    net.eval()
    out=net(imgtest)
    subfig.clear()
    subfig.imshow(out.cpu()[0][0].detach().numpy(), norm=colors.LogNorm(), cmap=mp);
    fig.savefig('Log_imgs/segimg_ep{}.jpg'.format(epoch))
    out=net(image)
    imgnp=out.cpu()[0][0].detach().numpy()
    fromarray(imgnp).save("pattern.tiff")
    out=net(imagesim)
    imgnp=out.cpu()[0][0].detach().numpy()
    fromarray(imgnp).save("patternsim.tiff")
    print('第{}轮结束'.format(epoch))
#plt.title("Training and Validation Loss")
#plt.plot(test_losses,label="val")
#plt.plot(train_losses,label="train")
#plt.xlabel("iterations")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()

