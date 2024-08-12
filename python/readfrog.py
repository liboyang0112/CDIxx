#!/usr/bin/env python
from imageIO import writeFloat,writePng
import numpy as np
#data = readImage("/home/boyang/SharedNTFS/images/einstein.png")
data = np.loadtxt("frog_trace_cov.dat",dtype=np.dtype("float32"))
#data = np.loadtxt("frog_trace.dat",dtype=np.dtype("float32"))
#data = np.loadtxt("Retr_Trace.dat",dtype=np.dtype("float32"))
#data = np.loadtxt("Retr_OrigTrace.dat",dtype=np.dtype("float32"))
data = data/np.max(data)
data=data.reshape((1024,1024))
print(data)
writeFloat("frogdata.bin",data)
cache = np.zeros((data.shape[0],data.shape[1],3), np.dtype("int8"))
writePng("out", data, cache, 0, 0)
writePng("out_log", data, cache, 1, 1)

