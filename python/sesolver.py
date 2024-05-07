import numpy
import matplotlib
from matplotlib import pyplot as plt
from math import cos, pi, exp, sin, sqrt

fig = plt.figure()
plt.ion()

length = 100
psi0=numpy.random.rand(length)-0.5
eigenVs=numpy.ones((length,length))
V=numpy.ones(length)
psi0 = psi0 / numpy.linalg.norm(psi0)
#for x in range(0, length):
#    psi0[x] = sin(6*pi*x/(length-1))
Hpsi = psi0
H2psi = psi0
mom=numpy.zeros(length)
norm = 100
deltaE = (30*pi*pi/length/length)/4
momcap = sqrt(2*deltaE)/(1+sqrt(2*deltaE))  # 5*sqrt(2)/length/(1+sqrt(2)/length)
niter = int(10/momcap)
print("iterations: ", niter)
snapshotInterval=100000  # int(niter/length)
print(snapshotInterval)
print(momcap)
beta = momcap
acc = 0.
V*=-4
#for x in range(0, length >> 8):
#    V[x << 8] = -4.01
#    V[(x << 8)+1] = -4.01
ieigen = 0
for x in range(0,niter):
    if x % snapshotInterval == 0 and ieigen<length-1:
        #print(ieigen)
        eigenVs[ieigen] = psi0
        ieigen+=1
    psi0 = H2psi+mom*(1-beta)
    #psi0 /= numpy.linalg.norm(psi0)
    Hpsi=2*psi0 + V*psi0
    for y in range(1, length):
        Hpsi[y] -= psi0[y-1]
        Hpsi[y-1] -= psi0[y]
    psi0 = H2psi
    H2psi=2*Hpsi + V*Hpsi
    for y in range(1, length):
        H2psi[y] -= Hpsi[y-1]
        H2psi[y-1] -= Hpsi[y]
    norm = numpy.linalg.norm(H2psi)
    H2psi /= norm
    #mom = H2psi + beta*psi0
    mom = (H2psi - psi0)
    beta = acc*(beta-momcap) + momcap
    if x % 1 == 0:
        #plt.plot(mom/numpy.linalg.norm(mom))
        plt.plot(psi0)
        plt.draw()
        plt.pause(0.01)
        fig.clear()
eigenVs[length-1] = H2psi+mom*(1-beta)
eigenVs[length-1] /= numpy.linalg.norm(eigenVs[length-1])

currentV = length-2
i = 2
plt.plot(eigenVs[length-1])
print("E(1)=%s\n"%(Hpsi.dot(psi0)))
while i < 5:
    for j in range(1,i):
        eigenVs[length-i] -= eigenVs[length-i].dot(eigenVs[length-i+j])*eigenVs[length-i+j]
    eigenVs[length-i] /= numpy.linalg.norm(eigenVs[length-i])
    print(eigenVs[length-i].dot(eigenVs[length-i+1]))
    overlap=eigenVs[length-i].dot(eigenVs[length-i+1])
    if abs(overlap) > 0.0001:
        currentV-=1
        eigenVs[length-i] = eigenVs[currentV]
        continue
    currentV-=1
    psi0 = eigenVs[length-i]
    Hpsi=2*psi0 + V*psi0
    for y in range(1, length):
        Hpsi[y] -= psi0[y-1]
        Hpsi[y-1] -= psi0[y]
    print("E(%s)=%s\n"%(i,Hpsi.dot(psi0)),end=' ')
    plt.plot(eigenVs[length-i])
    plt.draw()
    i+=1
plt.pause(100)
fig.clear()
print(Hpsi/eigenVs[length-1])
numpy.savetxt("b.txt", psi0)
