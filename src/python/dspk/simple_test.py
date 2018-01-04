import numpy as np
from dspk import dspk


#spiky data
spikes = np.ones([100,100,10])
spikes[5,5,2] = 10
spikes[5,5,3] = 25
spikes[25,25,5] = 1.5
spikes[55,55,7] = 10

(data,good_map,bad_pix_number) = dspk(spikes)
print(np.where(good_map==0))
print(bad_pix_number)


