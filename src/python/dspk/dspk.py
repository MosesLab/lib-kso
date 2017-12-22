import numpy as np
import tensorflow as tf

# The beginnings of a Tensor Flow based data despiking routine.  The goal is to despike an image cube efficiently using a
# GPU

def dspk(data):

    # contruct 3-D averaging kernel and form Tensor of shape (filter_depth, filter_height, filter_width, in_channel, out_channel)
    mu_kernel = np.ones([5,5,5])
    mu_kernel = np.expand_dims(mu_kernel,-1)
    mu_kernel = np.expand_dims(mu_kernel,-1)

    mu_krn = tf.convert_to_tensor(mu_kernel, dtype =np.float32)

    #form data cube into tensor of shape (batch, depth, height, width, channels)
    data = np.expand_dims(data,0)
    data = np.expand_dims(data,-1)
    dt= tf.convert_to_tensor(data,dtype=np.float32)

    #form a map of good and bad pixels witht the same dimensions as data and form Tensor
    goodmap = data*0+1
    goodmap = np.expand_dims(goodmap,0)
    goodmap = np.expand_dims(goodmap,-1)
    gm = tf.convert_to_tensor(goodmap,dtype=np.float32)

    #define a near-local smoothing kernel for replacing bad pixels


    #Calculate running normilization
    norm = tf.nn.conv3d(gm,mu_krn,padding = same, name = 'Running Normilization')

    #Calculate a neighborhood mean
    g_data = tf.multiply(gm,dt)
    n_mean = tf.nn.conv3d(g_data,mu_krn)
    n_mean = tf.divide(n_mean,norm)

    #Deviation
    dev = tf.subtract(g_data,n_mean)
    n_std = tf.nn.conv3d(dev^2,mu_krn)
    n_std = tf.divide(n_std,norm)

    #Compare pixel deviation to local standard deviation
    








