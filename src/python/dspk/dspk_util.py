import numpy as np

def flatten_cube(orig_data, sz_x, sz_y, sz_z):

    panes = int(np.sqrt(sz_z))
    orig_data_flat = np.reshape(orig_data, [sz_x, sz_y, panes, panes])
    orig_data_flat = np.swapaxes(orig_data_flat, 1, 2)
    orig_data_flat = np.swapaxes(orig_data_flat, 0, 1)
    orig_data_flat = np.swapaxes(orig_data_flat, 2, 3)
    orig_data_flat = np.reshape(orig_data_flat, [sz_x * panes, sz_y * panes])

    return orig_data_flat