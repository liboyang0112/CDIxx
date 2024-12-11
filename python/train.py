#!/usr/bin/env python
from os.path import exists
from os import mkdir
import torch
#from Deep import Deep
from torch import device, tensor, nn, _dynamo
from torch.nn import functional as F
from torch.utils.data import DataLoader
from imageIO import writeFloat,readImage,writePNG
import numpy as np
from torchvision.transforms import CenterCrop, Resize
from UNet import UNet
from unetDataLoader import unetDataLoader as ul
#from torch.utils.tensorboard import SummaryWriter
_dynamo.config.suppress_errors = True

net = UNet(1,4).cuda()
#net = Deep(1,7).cuda()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.01,betas = (0.9, 0.999))
schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6, eps=1e-8, threshold=1e-4)
#loss_func = nn.BCELoss()
#loss_func = nn.L1Loss()
loss_func = nn.MSELoss()
trainsz = 256
bs = 4
data = ul("./traindb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
datatest = ul("./testdb", 1, trainsz,trainsz, 1, trainsz,trainsz,device('cuda:0'))
dataloader = DataLoader(data, batch_size = bs, shuffle = True,num_workers = 0,drop_last = True)
testloader = DataLoader(datatest, batch_size = bs, shuffle = True,num_workers = 0,drop_last = True)
EPOCH = 300
print('load net')
testIdx = 1
ModelSave = 'model_4_2_linear'
if not exists(ModelSave):
    mkdir(ModelSave)
if not exists('Log_imgs'):
    mkdir('Log_imgs')
if exists(ModelSave + '/Unet.pt'):
    net.load_state_dict(torch.load(ModelSave + '/Unet.pt'))
#net = torch.compile(net)
print('load success')
imgval,label = datatest[testIdx]
cache = np.zeros((label.shape))
writePNG("Log_imgs/seglab.png", label.cpu()[0].numpy(), cache, 1, 1)
imgtest = torch.unsqueeze(imgval,dim = 0)
testimg, testlabel = next(iter(testloader))
train_losses = []
test_losses = []
runExp = 0
resize = Resize([trainsz,trainsz])
if exists('broad_pattern.bin'):
    runExp = 1
    image = tensor(np.asarray(readImage('broad_pattern.bin')))
    print(image.shape)
    x0 = (trainsz - image.shape[0]) >> 1
    y0 = (trainsz - image.shape[1]) >> 1
    #if x0 > 0:
    #image = F.pad(image,(100,100,100,100),"constant",0)
    #elif x0 < 0:
    crp = CenterCrop(trainsz)
    image = crp(image)
    image = image.clone().detach()
    image = image.view(1,1,image.shape[0], image.shape[1]).to(device('cuda:0'))
    image = resize(image)
lossfile = open(ModelSave+"/losses.txt", "a",)
bestloss = 1e10
for epoch in range(EPOCH):
    print('开始第{}轮'.format(epoch))
    total_loss = 0
    avgvalloss = 0
    net.train()
    valout = 0
    for i,(img,label) in enumerate(dataloader):
        trainimg = net(img)
        loss = loss_func(trainimg,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #train_losses.append(loss.item())
        loss = loss.item()
        if i%20 == 0:
            print("training loss = %f"%(loss / bs * 1e6))
        total_loss += loss
        valout = net(testimg)
        avgvalloss += loss_func(valout, testlabel).item()
        #test_losses.append(loss.item())
    print("validation loss = %f"%(avgvalloss * 1e6))
    print("total training loss = %f"%(total_loss * 1e6))
    #schedule.step(avgvalloss)
    #lr = optimizer.state_dict()['param_groups'][0]['lr']
    #print("lr=%f"%(lr))
    lossfile.write("%d %e %e\n"%(epoch, total_loss, avgvalloss))
    #if lr <= 1e-4:
    #    break
    bestloss = avgvalloss
    torch.save(net.state_dict(),ModelSave + '/Unet.pt')
    writePNG(f'Log_imgs/segimg_ep{epoch}.png',valout.cpu()[0][0].detach().numpy(),cache, 1)
    writePNG(f'Log_imgs/segimg_train_ep{epoch}.png',trainimg.cpu()[0][0].detach().numpy(),cache, 1)
    writePNG('Log_imgs/segimg_train_lab.png',trainimg.cpu()[0][0].detach().numpy(),cache, 1)
    net.eval()
    if runExp:
        out = net(image)
        imgnp = out.cpu()[0][0].detach().numpy()
        writeFloat("pattern.bin",imgnp)
        writePNG('Log_imgs/pattern.png',imgnp,cache, 1)
    print('第{}轮结束'.format(epoch))
lossfile.close()
