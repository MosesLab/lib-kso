import numpy as np

def flatten_cube(data_3D, sz_x, sz_y, sz_z):

    panes = int(np.sqrt(sz_z))
    orig_data_flat = np.reshape(data_3D, [sz_x, sz_y, panes, panes])
    orig_data_flat = np.swapaxes(orig_data_flat, 1, 2)
    orig_data_flat = np.swapaxes(orig_data_flat, 0, 1)
    orig_data_flat = np.swapaxes(orig_data_flat, 2, 3)
    data_3D_flat = np.reshape(orig_data_flat, [sz_x * panes, sz_y * panes])

    return data_3D_flat

def add_frame(data_3D, axes, f_sz=1):

    f = f_sz
    sz = np.shape(data_3D)

    i = 0
    for s in sz:    # iterate through shape since we don't know how big it is
        t = s - 1
        if i in axes:

            data = np.rollaxis(data_3D, i)   # access arbitrary dimension of the array

            print(data.shape)

            # Apply frame
            data[0:f,:,:] = 0
            data[s-f:s,:,:] = 0


            data_3D = np.rollaxis(data, np.mod(-i,2))  # undo roll

        i = i + 1   # keep track of our own looping variable


    return data_3D

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