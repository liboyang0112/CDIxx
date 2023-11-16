#!/usr/bin/env python
from libimageIO_cython import writeFloat,readImage,writePng
import numpy as np
data = readImage("/home/boyang/SharedNTFS/images/einstein.png")
writeFloat("einstein.bin",data)
data = readImage("einstein.bin")
cache = np.zeros((data.shape[0],data.shape[1],3), np.int8())
writePng("out", data, cache, 0, 0)

