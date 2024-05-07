import time
import numpy
from matplotlib import pyplot as plt
from math import cos, pi, exp, sin

fig = plt.figure()
plt.ion()

length = 100
psi0=numpy.random.rand(length)
V=numpy.ones(length)
psi0 = psi0 / numpy.linalg.norm(psi0)
psiprev = psi0
psiprevprev = psi0
#for x in range(0, length):
#    psi0[x] = sin(6*pi*x/(length-1))
niter = 300
Hpsi = psi0
H2psi = psi0
mom=numpy.zeros(length)
norm = 100
alpha = 1.0
V*=-4
#for x in range(0, length >> 4):
#    V[x << 4] = -4.1
#    V[(x << 4)+1] = -4.1
for x in range(0,niter):
    if x % 1 == 0:
        plt.plot(psi0)
        plt.draw()
        plt.pause(0.01)
        fig.clear()
    Hpsi=2*psi0 + V*psi0
    for y in range(1, length):
        Hpsi[y] -= psi0[y-1]
        Hpsi[y-1] -= psi0[y]
    H2psi=2*Hpsi + V*Hpsi
    for y in range(1, length):
        H2psi[y] -= Hpsi[y-1]
        H2psi[y-1] -= Hpsi[y]
    E2 = H2psi.dot(psi0)
    print(E2)
    psiprevprev=psiprev
    psiprev=psi0
    psi0 = H2psi-E2*psi0-psiprevprev
    psi0 /= numpy.linalg.norm(psi0)
print(Hpsi/psi0)
numpy.savetxt("b.txt", psi0)
