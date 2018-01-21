import numpy as np

import kso.cpp.img.dspk.dspk as despike

A = np.empty([2,3,4], dtype=np.float32)

despike.remove_noise_3D(A)


