#!/usr/bin/env python

import numpy as np
from readHj import runFGA

#data from NP
Example_variables = np.load('Example_whitelight_experiment.npz')
B = Example_variables['B']
spect_I = Example_variables['spect_I']
spect_lambda = Example_variables['spect_lambda']
runFGA(B,spect_lambda / spect_lambda[0],spect_I)
