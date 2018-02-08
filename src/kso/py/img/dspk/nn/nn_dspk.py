
import numpy as np
import matplotlib.pyplot as plt

import kso.cuda.instrument.IRIS as iris

from kso.py.tools.scroll_stepper import IndexTracker

from keras.callbacks import TensorBoard

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import adam
from keras.optimizers import Nadam
from keras.utils import plot_model
from keras import regularizers
from keras.initializers import RandomNormal
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

bsz_x = 1024
bsz_y = 1024
bsz_z = 1024

data = np.empty([bsz_z, bsz_y, bsz_x], dtype=np.float32)
nsz = np.empty(3, dtype=np.uint)
iris.read_fits_raster_ndarr(data, nsz)
sz_x = int(nsz[2])
sz_y = int(nsz[1])
sz_z = int(nsz[0])
sz3 = sz_x * sz_y * sz_z
data = np.resize(data, [sz_z, sz_y, sz_x])

x = np.arange(0, sz_x)
y = np.arange(0, sz_y)
z = np.arange(0, sz_z)

xx, yy, zz = np.meshgrid(x,y,z)

print(xx.shape)


net = Sequential()

init = RandomNormal(mean=0.0, stddev=2e-3, seed=None)


l1_units = 1024
net.add(Dense(l1_units, activation='tanh', input_shape=(3,)))

l2_units = 1024
net.add(Dense(l2_units, activation='tanh'))

l3_units = 1
net.add(Dense(l3_units, activation=None))

sgd = SGD(lr=4e-4, decay=1e-4, momentum=0.9, nesterov=True)

# Compile parameters into the model
net.compile(optimizer=sgd, loss='mse')

xx_f = xx.flatten()
yy_f = yy.flatten()
zz_f = zz.flatten()
dt_f = data.flatten()
input = np.stack([xx_f,yy_f,zz_f], axis=1)
truth = np.expand_dims(dt_f,axis=-1)
print(input.shape)
print(truth.shape)

net.fit(input, truth, batch_size=128*1024,verbose=1, shuffle='batch')

net.save('model.h5')