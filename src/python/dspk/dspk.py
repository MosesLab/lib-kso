import numpy as np
import tensorflow as tf

# The beginnings of a Tensor Flow based data despiking routine.  The goal is to despike an image cube efficiently using a
# GPU

def dspk(data):

    #build Tensor Flow Graph
    sess = tf.Session()
    # contruct 3-D averaging kernel and form Tensor of shape (filter_depth, filter_height, filter_width, in_channel, out_channel)
    mu_kernel = np.ones([3,3,3])
    mu_kernel = np.expand_dims(mu_kernel,-1)
    mu_kernel = np.expand_dims(mu_kernel,-1)

    mu_krn = tf.convert_to_tensor(mu_kernel, dtype =np.float32)

    #form data cube into tensor of shape (batch, depth, height, width, channels)
    data = np.expand_dims(data,0)
    data = np.expand_dims(data,-1)

    dt= tf.convert_to_tensor(data,dtype=np.float32)






    #form a map of good and bad pixels witht the same dimensions as data and form Tensor
    goodmap = data*0+1
    gm = tf.convert_to_tensor(goodmap,dtype=np.float32)

    #initilize bad pixel count to zero
    bad_num = tf.constant(0,dtype=tf.float32)
    new_bad = tf.constant(0,dtype=tf.float32)

    #Function to identify bad pixels
    def identify_bad_pix(gm,mu_krn,dt,bad_num,new_bad):


        #Calculate running normilization
        norm = tf.nn.conv3d(gm,mu_krn,strides=[1,1,1,1,1],padding = "SAME")

        #Calculate a neighborhood mean
        g_data = tf.multiply(gm,dt)
        n_mean = tf.nn.conv3d(g_data,mu_krn,strides=[1,1,1,1,1],padding = "SAME")
        n_mean = tf.divide(n_mean,norm)

        #Deviation
        dev = tf.subtract(g_data,n_mean)
        n_std = tf.nn.conv3d(tf.square(dev),mu_krn,strides=[1,1,1,1,1],padding="SAME")
        n_std = tf.sqrt(tf.divide(n_std,norm))

        #Compare pixel deviation to local standard deviation
        sigmas = tf.constant(4.5,dtype=tf.float32)

        test = tf.multiply(sigmas,n_std)

        bad = tf.where(tf.greater(dev,test),tf.ones_like(dt),tf.zeros_like(dt))

        new_bad = tf.subtract(tf.reduce_sum(bad),bad_num)


        #update good map and count bad pixels
        gm = tf.subtract(gm,bad)
        bad_num = tf.add(bad_num,new_bad)
        print('test')
        return (gm,mu_krn,dt,bad_num,new_bad)

    #setup while loop convergence condition
    def end_bad_pix_search(gm,mu_krn,dt,bad_num,new_bad):
        return tf.less_equal(tf.divide(new_bad,bad_num),.01)

    find_bad_pix = tf.while_loop(end_bad_pix_search,identify_bad_pix,(gm,mu_krn,dt,bad_num,new_bad))
    list= sess.run(find_bad_pix)


    # (gm, mu_krn, dt, bad_num, new_bad) = identify_bad_pix(gm,mu_krn,dt,bad_num,new_bad)






    return list

    # define a near-local smoothing kernel for replacing bad pixels






