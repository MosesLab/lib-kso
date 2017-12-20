
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from scipy import signal

def gauss(M_x, M_y, std_x, std_y):
    n_x = np.arange(0, M_x) - (M_x - 1.0) / 2.0
    n_y = np.arange(0, M_y) - (M_y - 1.0) / 2.0
    sig2_x = 2 * std_x * std_x
    sig2_y = 2 * std_y * std_y
    w_x = np.exp(-n_x ** 2 / sig2_x)
    w_y = np.exp(-n_y ** 2 / sig2_y)

    w_x = np.reshape(w_x,[-1,1])
    w_y = np.reshape(w_y, [-1, 1])

    w_y = np.transpose(w_y)

    return w_x * w_y


g = gauss(10,10,5,5)

print(g)

plt.imshow(g)
plt.show(4)


