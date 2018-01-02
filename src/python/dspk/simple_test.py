import numpy as np
from dspk import dspk


spikes = np.ones([64,1024,1024])

spikes[10,10,10] = 10
spikes[10,10,20] = 10
spikes[10,10,30] = 10
spikes[10,10,43] = 10

gm = dspk(spikes)
print(np.where(spikes > 1))




