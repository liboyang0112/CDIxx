#!/usr/bin/env python
from sys import argv
import numpy as np
data = np.loadtxt(argv[1],dtype=np.dtype("float32"))
psnr=data[:,1]
ssim=data[:,0]
print(np.mean(psnr),np.std(psnr), np.mean(ssim),np.std(ssim))
