#!/usr/bin/env python
from imageIO import writeFloat,readImage,writePNG
import numpy as np
data = readImage("/home/boyang/SharedNTFS/images/cup/einstein.png")
writeFloat("einstein.bin",data)
data = readImage("einstein.bin")
cache = np.zeros((data.shape[0],data.shape[1],3), np.int8)
writePNG("out.png", data, cache, 0, 0)

