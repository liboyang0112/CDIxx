import time
import numpy
from matplotlib import pyplot as plt
from math import cos, pi, exp, sin

fig = plt.figure()
plt.ion()

length = 400
psi0=numpy.random.rand(length)
V=numpy.ones(length)
psi0 = psi0 / numpy.linalg.norm(psi0)
psiprev = psi0
#for x in range(0, length):
#    psi0[x] = sin(6*pi*x/(length-1))
niter = 300
Hpsi = psi0
H2psi = psi0
mom=numpy.zeros(length)
norm = 100
alpha = 1
V*=-4
for x in range(0, length >> 8):
    V[x << 8] = -4.01
    V[(x << 8)+1] = -4.01
for x in range(0,niter):
    psi0 = H2psi+mom*alpha
    psiprev = H2psi
    psi0 /= numpy.linalg.norm(psi0)
    Hpsi=2*psi0 + V*psi0
    for y in range(1, length):
        Hpsi[y] -= psi0[y-1]
        Hpsi[y-1] -= psi0[y]
    H2psi=2*Hpsi + V*Hpsi
    for y in range(1, length):
        H2psi[y] -= Hpsi[y-1]
        H2psi[y-1] -= Hpsi[y]
    H2psi /= numpy.linalg.norm(H2psi)
    mom = (H2psi - psiprev)*0.985
    if x % 1 == 0:
        #plt.plot(mom*1000)
        plt.plot(psi0)
        plt.draw()
        plt.pause(0.01)
        fig.clear()
#print(Hpsi/psi0)
print("E=",Hpsi.dot(psi0))
numpy.savetxt("b.txt", psi0)
