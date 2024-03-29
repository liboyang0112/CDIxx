import math
from math import cos, sin

qe = 1.6e-19
c=3e8
h=6.62607015e-34
#wl = 0.2e-9
ev = 50
wl = c*h/(ev*qe)
print("wave length:",wl*1e9,"nm")

#wl = 3e-9
#ev = c*h/(wl*qe)
#print(ev)

#Silicon @ 50eV
#rho = 2.32e6 #g/m3
#A = 28.1
#f1 = 3.11054
#f2 = 3.1003E-01
#d = 5e-6

#Tungsten @ 6.2keV
#rho = 19.3e6 #g/m3
#A = 183.85
#f1 = 70.7
#f2 = 8
#d = 1e-6

#Copper @ 3nm
#rho = 8.94e6 #g/m3
#A = 64
#f1 = 17.8
#f2 = 5.68
#d = 1e-8

#Iron @ 3nm
#rho = 7.86e6 #g/m3
#A = 55.847
#f1 = 16
#f2 = 7.145
#d = 1e-7

#Aluminium @ 533eV
#rho = 2.694e6 #g/m3
#A = 13
#f1 = 2
#f2 = 1.98
#d = 1e-6

#SiO2
#rho = 2.5e6 #g/m3
#A = 60
#@533eV
#f1 = 2
#f2 = 3.2
#@53eV
#f1 = 2
#f2 = 6.557390
#d = 1e-6

#Be @ 533eV
#rho = 1.845e6 #g/m3
#A = 9
#f1 = 2
#f2 = 0.50
#d = 1e-6
#
#MgO @ 533eV
#rho = 3.58e6 #g/m3
#A = 40
#f1 = 2
#f2 = 1.611
#d = 1e-5

#Si3N4 @ 533eV
rho = 3.12e6 #g/m3
A = 140.283
#@ 533eV
#f1 = 6
#f2 = 20.2
#@ 50eV
f1 = 2
f2 = 9.174170
d = 1e-6


#Carbon @ 533eV
#rho = 2.26e6 #g/m3
#A = 12
#f1 = 6
#f2 = 1.8212
#d = 3.5e-7

#Water @ 533eV
#rho = 1e6 #g/m3
#A = 18.0154
#f1 = 2
#f2 = 0.22
#d = 1e-5


NA = 6.022e23
re = 2.81e-15
n = rho*NA/A
dn = re*wl*f1*n
dk = re*wl*f2*n
dphi = dn*d
decay = math.exp(-2*dk*d)
print("atom number density = ", n)
print("Delta n", dn*wl/2/math.pi)
print("Delta phi = ", dphi)
print("attenuation", decay)

#for x in range(0,1000):
#  d = 1e-8*x
#  phi = dn*d
#  r = math.exp(-2*dk*d)
#  print(r*cos(phi), r*sin(phi))
