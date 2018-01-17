import numpy as np


# original size of the array
os_t = 64
os_y = os_t
os_l = os_t

# size of the kernel
ks_t = 5
ks_y = 5
ks_l = 5

# Construct test array
A = np.arange(0, os_t, dtype=np.float32)
A = np.expand_dims(A,-1)
A = np.expand_dims(A,-1)
A = np.tile(A, [1, os_y, os_l])

# new size of array after striding
ns_t = ks_t
ns_y = os_y
ns_l = os_l
ns_T = ns_t / np.ceil(ks_t / 2)

# size of strides
ms_T = A.strides[0] * 5
ms_t = A.strides[0]
ms_y = A.strides[1]
ms_l = A.strides[2]

B = np.lib.stride_tricks.as_strided(A, shape=[ns_T, ns_t, ns_y, ns_l], strides=[ms_T, ms_t, ms_y, ms_l])

print(B)