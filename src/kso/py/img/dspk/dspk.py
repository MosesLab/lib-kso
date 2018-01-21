import numpy as np

import kso.cpp.img.dspk.dspk as despike

b = 2

A = np.arange(0.1,3,0.1, dtype=np.float32)

despike.remove_noise_3D(A)
