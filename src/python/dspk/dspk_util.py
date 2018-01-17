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