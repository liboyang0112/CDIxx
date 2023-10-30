from libimageIO_cython import writeFloat,readImage
import matplotlib.pyplot as plt
row, col, data = readImage("/home/boyang/SharedNTFS/images/einstein.png")
writeFloat("einstein.bin",data);
row, col, data = readImage("einstein.bin")
plt.imsave("out.png",data);

