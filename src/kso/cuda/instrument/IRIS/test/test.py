
import numpy as np
import matplotlib.pyplot as plt

import kso.cuda.instrument.IRIS as iris

from kso.py.tools.scroll_stepper import IndexTracker

bsz_x = 1024
bsz_y = 1024
bsz_z = 1024

arr = np.empty([bsz_z, bsz_y, bsz_x], dtype=np.float32)
nsz = np.empty(3, dtype=np.uint)

iris.read_fits_raster_ndarr(arr, nsz)

print(nsz)

sz_x = int(nsz[2])
sz_y = int(nsz[1])
sz_z = int(nsz[0])



sz3 = sz_x * sz_y * sz_z
arr = np.resize(arr, [sz_z, sz_y, sz_x])

# arr_slice = arr[25,:,:]

fig, ax = plt.subplots(1, 1)
tracker = IndexTracker(ax, arr, 0)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#
# plt.imshow(arr_slice)
#
plt.show()