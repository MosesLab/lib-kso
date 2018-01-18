import numpy as np

def time_stride_5D(A, krn_sz_t, stride_sz_t):



    if len(A.shape) != 5: raise ValueError('Number of dimensions of input array must be 5.')
    os_b = A.shape[0]
    os_t = A.shape[1]
    os_y = A.shape[2]
    os_l = A.shape[3]
    os_c = A.shape[4]

    if os_b != 1: raise  ValueError('First dimension must be a singleton dimension')


    ks_t = krn_sz_t


    # new size of array after striding
    ns_t = stride_sz_t
    ns_y = os_y
    ns_l = os_l
    ns_c = os_c

    ks_t_2 = np.int(np.floor(ks_t / 2))
    es_b = ns_t - ks_t_2
    ns_b = os_b * np.int(np.floor((os_t - ks_t_2) / es_b))
    print(ns_b)

    print(A.strides)

    # size of strides
    ms_t = A.strides[1]
    ms_y = A.strides[2]
    ms_l = A.strides[3]
    ms_b = ms_t * es_b
    ms_c = A.strides[4]

    B = np.lib.stride_tricks.as_strided(A, shape=[ns_b, ns_t, ns_y, ns_l, ns_c], strides=[ms_b, ms_t, ms_y, ms_l, ms_c])

    return B

def main():

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


if __name__ == "__main__": main()