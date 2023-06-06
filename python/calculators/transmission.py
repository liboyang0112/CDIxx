import math

qe = 1.6e-19
c=3e8
h=6.62607015e-34
#wl = 0.2e-9
ev = 533
wl = c*h/(ev*qe)
print(wl)

#wl = 3e-9
#ev = c*h/(wl*qe)
#print(ev)

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

#Water @ 533eV
#rho = 1e6 #g/m3
#A = 18.0154
#f1 = 2
#f2 = 0.22
#d = 1e-5

#Aluminium @ 533eV
#rho = 2.694e6 #g/m3
#A = 13
#f1 = 2
#f2 = 1.98
#d = 1e-6

#SiO2 @ 533eV
#rho = 2.5e6 #g/m3
#A = 60
#f1 = 2
#f2 = 3.2
#d = 0.13e-3

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

#Carbon @ 533eV
rho = 2.26e6 #g/m3
A = 12
f1 = 6
f2 = 1.8212
d = 2e-6

NA = 6.022e23
re = 2.81e-15
n = rho*NA/A
dn = re*wl*f1*n
dk = re*wl*f2*n
dphi = dn*d
decay = math.exp(-2*dk*d)
print("atmp number density = ", n)
print("Delta n", dn*wl/2/math.pi)
print("Delta phi = ", dphi)
print("attenuation", decay)
