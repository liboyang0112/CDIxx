#!/usr/bin/env python
import torch
from os.path import exists
from os import mkdir
#from mUNet import mUNet
from UNet import UNet
from torch import device, tensor, nn
from unetDataLoader import unetDataLoader as ul
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from libimageIO_cython import writeFloat,readImage,writePng
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import CenterCrop

#writer = SummaryWriter()
#array = tensor([float(1)]).to(device('cuda:0'))
#net = mUNet(array,channels = 1).cuda()
net = UNet(channels = 1).cuda()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.0001,betas = (0.9, 0.999))
#loss_func = nn.BCELoss()
#loss_func = nn.L1Loss()
loss_func = nn.MSELoss()
trainsz = 256
data = ul("./traindb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
datatest = ul("./testdb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
dataloader = DataLoader(data, batch_size = 4, shuffle = True,num_workers = 0,drop_last = True)
testloader = DataLoader(datatest, batch_size = 4, shuffle = True,num_workers = 0,drop_last = True)
EPOCH = 1000
print('load net')
testIdx = 1
ModelSave = 'model_4_2'
if not exists(ModelSave):
    mkdir(ModelSave)
if not exists('Log_imgs'):
    mkdir('Log_imgs')
if exists(ModelSave + '/Unet.pt'):
    net.load_state_dict(torch.load(ModelSave + '/Unet.pt'))
print('load success')
imgval,label = datatest[testIdx]
cache = np.zeros((label.shape))
writePng("Log_imgs/seglab.png", label.cpu()[0].numpy(), cache, 1, 1)
imgtest = torch.unsqueeze(imgval,dim = 0)
testimg, testlabel = next(iter(testloader))
train_losses = []
test_losses = []

image = readImage('broad_pattern.bin')
x0 = (trainsz - image.shape[0]) >> 1
y0 = (trainsz - image.shape[1]) >> 1
image = tensor(np.asarray(image))
if x0 > 0:
    image = F.pad(image,(x0,x0,y0,y0),"constant",0)
elif x0 < 0:
    crp = CenterCrop(trainsz)
    image = crp(image)

#transform = T.Resize(trainsz)
#image = tensor(image)
#image = transform(image)
#image = np.asarray(image)
image = image.clone().detach()
image = image.view(1,1,trainsz,trainsz).to(device('cuda:0'))

#imagesim = readImage('simfloat.bin')
#x0 = (trainsz-imagesim.shape[0])>>1
#y0 = (trainsz-imagesim.shape[1])>>1
#
#imagesim = tensor(np.asarray(imagesim))
#imagesim = torch.nn.functional.pad(imagesim,(x0,x0,y0,y0),"constant",0)
#imagesim = imagesim.view(1,1,trainsz,trainsz).to(device('cuda:0'))
for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    net.train()
    total_loss = 0
    for i,(img,label) in enumerate(dataloader):
        loss = loss_func(net(img),label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #train_losses.append(loss.item())
        if i%20 == 0:
            print("training loss = %f"%(loss.item() * 1e6))
        total_loss  += loss.item() * 2e3
        loss = loss_func(net(testimg), testlabel)
        #test_losses.append(loss.item())
        if i%20 == 0:
            print("validation loss = %f"%(loss.item() * 1e6))
    print("total training loss = %f"%total_loss)
    torch.save(net.state_dict(),ModelSave + '/Unet.pt')
    net.eval()
    out = net(imgtest)
    writePng('Log_imgs/segimg_ep{}.png'.format(epoch),out.cpu()[0][0].detach().numpy(),cache, 1)
    out = net(image)
    imgnp = out.cpu()[0][0].detach().numpy()
    writeFloat("pattern.bin",imgnp)
    #out = net(imagesim)
    #imgnp = out.cpu()[0][0].detach().numpy()
    #writeFloat("patternsim.bin",fromarray(imgnp))
    print('第{}轮结束'.format(epoch))
#plt.title("Training and Validation Loss")
#plt.plot(test_losses,label = "val")
#plt.plot(train_losses,label = "train")
#plt.xlabel("iterations")
#plt.ylabel("Loss")
#plt.legend()
#plt.show()

