#!/usr/bin/env python
from torch import utils, tensor
from unetDataLoader import cythonLoader

class unetDataLoader(utils.data.Dataset):
    def __init__(self, db_path, chan, row, col, chanl, rowl, coll, device, transform=None, target_transform=None):
        self.loader = cythonLoader(db_path, chan, row, col, chanl, rowl, coll);
        self.path = db_path
        self.chan = chan
        self.row = row
        self.col = col
        self.chanl = chanl
        self.rowl = row
        self.coll = coll
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        imgnp, labnp = self.loader.__getitem__(index);
        imgnp.shape = (self.chan, self.row, self.col);
        labnp.shape = (self.chanl, self.rowl, self.coll);
        #img = tensor(imgnp).to(self.device).reshape([self.chan, self.row, self.col]);
        #lab = tensor(labnp).to(self.device).reshape([self.chanl, self.rowl, self.coll]);
        img = tensor(imgnp).to(self.device);
        lab = tensor(labnp).to(self.device);
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lab = self.target_transform(lab)
        return img,lab
    def __len__(self):
        return self.loader.length
    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.path + ')'
