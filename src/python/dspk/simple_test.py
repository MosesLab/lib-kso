import numpy as np
from dspk import dspk


spikes = np.ones([100,1024,1024])

spikes[10,10,10] = 100
spikes[10,10,20] = 100
spikes[10,10,30] = 100
spikes[10,10,43] = 100

gm = dspk(spikes)
gm = np.squeeze(gm)
print(gm.shape)
print([gm[3],gm[4]])




