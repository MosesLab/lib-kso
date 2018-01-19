import numpy as np

import shaping

def test_time_stride_5D():

    # original size of the array
    os_t = 16
    os_y = 3
    os_l = 3

    # size of the kernel
    ks_t = 5

    # size of stride
    ss_t = 7

    # Construct test array
    A = np.arange(0, os_t, dtype=np.int)
    A = np.expand_dims(A, -1)
    A = np.expand_dims(A,-1)
    A = np.tile(A, [1, os_y, os_l])
    A = np.expand_dims(A,0)
    A = np.expand_dims(A,-1)

    # print(A)

    B = time_stride_5D(A, ks_t, ss_t)

    print(B)
