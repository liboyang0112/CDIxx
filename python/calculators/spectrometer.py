lambdamin = 40e-9
lambdamax = 200e-9
distance = 0.2
detector_res = 50e-6
detector_npix = 1024
sample_size = 20e-6
oversample = lambdamin*distance/detector_res/sample_size
#sample_size = lambdamin*distance/detector_res/oversample
spacing = 2*lambdamax*distance/detector_npix/detector_res
npoints = sample_size / spacing
print(f"sample_size={sample_size*1e6:.2f}um")
print(f"spacing={spacing*1e6:.2f}um")
print(f"oversampling ratio={oversample}")
print(f"number of points={npoints:.2f}")


