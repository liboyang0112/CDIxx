#!/usr/bin/env python
import torch
from os.path import exists
from os import mkdir
from model import UNet
from libunetDataLoader_cython import unetDataLoader as ul
from torch import utils, device, tensor, nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

net = UNet(channels=1).cuda()
optimizer = torch.optim.Adam(net.parameters())
loss_func = nn.BCELoss()
data = ul("./testdb", 1, 256,256, 1, 256,256,device('cuda:0'))
dataloader = DataLoader(data, batch_size=4, shuffle=True,num_workers=0,drop_last=True)
EPOCH=1000
print('load net')
if not exists('SAVE'):
    mkdir('SAVE')
if not exists('Log_imgs'):
    mkdir('Log_imgs')
if exists('SAVE/Unet.pt'):
    net.load_state_dict(torch.load('SAVE/Unet.pt'))
print('load success')
for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    net.train()
    for i,(img,label) in  enumerate(dataloader):
        img_out=net(img)
        loss=loss_func(img_out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(net.state_dict(),r'SAVE/Unet.pt')
    img,label=data[90]
    img=torch.unsqueeze(img,dim=0)
    net.eval()
    out=net(img)
    plt.imsave('Log_imgs/segimg_ep{}_90th_pic.jpg'.format(epoch), out.cpu()[0][0].detach().numpy())
    print('第{}轮结束'.format(epoch))

