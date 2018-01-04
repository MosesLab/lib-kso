import numpy as np
from dspk import dspk


spikes = np.ones([10,10,10])

spikes[5,5,5] = 100

list = dspk(spikes)
print(list)


