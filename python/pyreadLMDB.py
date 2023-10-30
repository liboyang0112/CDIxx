#!/usr/bin/env python
from unetDataLoader import unetDataLoader as ul
from torch import utils, device, tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

a = ul("./testdb", 256,256,256,256,device('cuda:0'))
data_train = DataLoader(a, batch_size=4, shuffle=True)
for batch_idx, samples in enumerate(data_train):
    print(batch_idx,samples)
#img, lab = a.__getitem__(990)
#plt.imsave("test.png",lab.cpu().numpy())
