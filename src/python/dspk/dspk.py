import numpy as np
import tensorflow as tf

# The beginnings of a Tensor Flow based data despiking routine.  The goal is to despike an image cube efficiently using a
# GPU

def dspk(data):
    # build Tensor Flow Graph
    sess = tf.Session()

    # contruct 3-D averaging kernel and form Tensor of shape (filter_depth, filter_height, filter_width, in_channel, out_channel)
    mu_kernel = np.ones([5, 5, 5])
    mu_kernel = np.expand_dims(mu_kernel, -1)
    mu_kernel = np.expand_dims(mu_kernel, -1)

    mu_krn = tf.convert_to_tensor(mu_kernel, dtype=np.float32)

    # form data cube into tensor of shape (batch, depth, height, width, channels)
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, -1)

    dt = tf.convert_to_tensor(data, dtype=np.float32)
    # #try using a placeholder node for large arrays
    # dt = tf.placeholder(dtype=tf.float32)

    # form a map of good and bad pixels witht the same dimensions as data and form Tensor
    goodmap = data * 0 + 1
    gm = tf.convert_to_tensor(goodmap, dtype=np.float32)

    # initilize bad pixel count to zero
    bad_num = tf.constant(0, dtype=tf.float32)
    # new_bad = tf.constant(0, dtype=tf.float32)
    percent_change = tf.constant(1, dtype=tf.float32)

    #count while loop iterations
    i = tf.constant(0)



    # Function to identify bad pixels
    def identify_bad_pix(gm, mu_krn, dt, bad_num, percent_change,i):
        # Calculate running normilization
        norm = tf.nn.conv3d(gm, mu_krn, strides=[1, 1, 1, 1, 1], padding="SAME")

        # Calculate a neighborhood mean
        g_data = tf.multiply(gm, dt)
        n_mean = tf.nn.conv3d(g_data, mu_krn, strides=[1, 1, 1, 1, 1], padding="SAME")
        n_mean = tf.divide(n_mean, norm)

        # Deviation
        dev = tf.subtract(g_data, n_mean)
        n_std = tf.nn.conv3d(tf.square(dev), mu_krn, strides=[1, 1, 1, 1, 1], padding="SAME")
        n_std = tf.sqrt(tf.divide(n_std, norm))

        # Compare pixel deviation to local standard deviation
        sigmas = tf.constant(4.5, dtype=tf.float32)

        test = tf.multiply(sigmas, n_std)

        bad = tf.where(tf.greater(dev, test), tf.ones_like(dt), tf.zeros_like(dt))

        new_bad = tf.reduce_sum(bad)
        percent_change = tf.divide(new_bad, bad_num)

        # update good map and count bad pixels
        gm = tf.subtract(gm, bad)
        bad_num = tf.add(bad_num, new_bad)
        i = tf.add(i,1)

        return (gm, mu_krn, dt, bad_num, percent_change,i)

    # setup while loop convergence condition
    def end_bad_pix_search(gm, mu_krn, dt, bad_num, percent_change,i):
        return tf.greater(percent_change, .01)

    (gm, mu_krn, dt, bad_num, percent_change, i) = tf.while_loop(end_bad_pix_search, identify_bad_pix,
                                 loop_vars=(gm, mu_krn, dt, bad_num, percent_change,i))



    # (gm, mu_krn, dt, bad_num, percent_change,i) = sess.run(find_bad_pix)







    # define a near-local smoothing kernel for replacing bad pixels
    skern_size = 5
    smoothing_kernel = np.empty([skern_size,skern_size,skern_size])
    for i in range(skern_size):
        for j in range(skern_size):
            for k in range (skern_size):
                r = np.sqrt((i-2)**2+(j-2)**2+(k-2)**2)
                smoothing_kernel[i,j,k] = np.exp((-r)/(1+r**3))

    smoothing_kernel = np.expand_dims(smoothing_kernel,-1)
    smoothing_kernel = np.expand_dims(smoothing_kernel,-1)
    smoothing_krn = tf.convert_to_tensor(smoothing_kernel,dtype=np.float32)

    #smooth data
    g_data = tf.multiply(gm, dt)
    norm = tf.nn.conv3d(gm, smoothing_krn, strides=[1, 1, 1, 1, 1], padding="SAME")
    dt = tf.nn.conv3d(g_data,smoothing_krn,strides=[1,1,1,1,1],padding='SAME')
    dt = tf.divide(dt,norm)


    #evaluate what we need
    (dt,gm,bad_num) = sess.run([dt,gm,bad_num])

    print('Number of iterations', i) #why is this always 4?
    print('Bad Pix Found',bad_num)
    return (np.squeeze(dt), np.squeeze(gm), bad_num)






