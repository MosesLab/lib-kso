import numpy as np

import matplotlib.pyplot as plt

def synth_iris(y_sz = 1024, l_sz = 256, t_sz=1):

    y = np.arange(0,1,y_sz,dtype=np.float32)
    l = np.arange(0,1,l_sz, dtype=np.float32)

    y = np.expand_dims(y,-1)
    l = np.expand_dims(l, 0)

