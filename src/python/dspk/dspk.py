import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug


# The beginnings of a Tensor Flow based data despiking routine.  The goal is to despike an image cube efficiently using a
# GPU

def dspk(data, std_dev=4.5, Niter=10):



    # contruct 3-D averaging kernel and form Tensor of shape (filter_depth, filter_height, filter_width, in_channel, out_channel)
    mu_kernel = np.ones([5, 5, 5])
    mu_kernel = np.expand_dims(mu_kernel, -1)
    mu_kernel = np.expand_dims(mu_kernel, -1)

    mu_krn = tf.convert_to_tensor(mu_kernel, dtype=np.float32)

    # form data cube into tensor of shape (batch, depth, height, width, channels)
    t_sz = 32
    data = np.expand_dims(data, 0)
    data = np.expand_dims(data, -1)
    splits = np.arange(0, data.shape[1], )
    split_data =

    # dt = tf.convert_to_tensor(data, dtype=np.float32)
    dt = tf.placeholder(np.float32, [1, None, None, None, 1])


    # form a map of good and bad pixels witht the same dimensions as data and form Tensor
    goodmap = data * 0 + 1
    gm = tf.convert_to_tensor(goodmap, dtype=np.float32, name='gm')

    # initilize bad pixel count to zero
    bad_num = tf.constant(0, dtype=tf.float32, name='bad_num')
    # new_bad = tf.constant(0, dtype=tf.float32)
    percent_change = tf.constant(1, dtype=tf.float32, name='percent_change')

    # count while loop iterations
    index = tf.constant(0, name='index')

    # Function to identify bad pixels
    def identify_bad_pix(gm, mu_krn, dt, bad_num, percent_change, index, Niter):


        # Calculate running normalization
        norm = tf.nn.conv3d(gm, mu_krn, strides=[1, 1, 1, 1, 1], padding="SAME", name="norm")

        # Calculate a neighborhood mean
        g_data = tf.multiply(gm, dt, name='g_data')
        n_mean = tf.nn.conv3d(g_data, mu_krn, strides=[1, 1, 1, 1, 1], padding="SAME", name='n_mean')
        n_mean = tf.divide(n_mean, norm, name='n_mean_norm')


        # Deviation
        dev = tf.subtract(dt, n_mean, name='dev')
        g_dev = tf.multiply(gm, dev, name='g_dev')
        g_var = tf.square(g_dev, name='g_var')
        n_std = tf.nn.conv3d(g_var, mu_krn, strides=[1, 1, 1, 1, 1], padding="SAME")

        n_std = tf.sqrt(tf.divide(n_std, norm), name='n_std_norm')

        # Compare pixel deviation to local standard deviation
        sigmas = tf.constant(std_dev, dtype=tf.float32, name='sigmas')

        test = tf.multiply(sigmas, n_std, name='test')

        bad = tf.where(tf.greater(g_dev, test), tf.ones_like(dt), tf.zeros_like(dt), name='bad')


        new_bad = tf.reduce_sum(bad, name='new_bad')
        percent_change = tf.divide(new_bad, bad_num, name='percent_change')

        # update good map and count bad pixels
        gm = tf.subtract(gm, bad, name='gm')
        bad_num = tf.add(bad_num, new_bad, name='bad_num')
        index = tf.add(index, 1, name='index')

        return (gm, mu_krn, dt, bad_num, percent_change, index, Niter)

    # setup while loop convergence condition
    def end_bad_pix_search(gm, mu_krn, dt, bad_num, percent_change, index, Niter):
        perc = tf.greater(percent_change, 1e-6)
        iter = tf.less(index, Niter)
        return  tf.logical_and(perc, iter)

    (gm, mu_krn, dt, bad_num, percent_change, index, Niter) = tf.while_loop(end_bad_pix_search, identify_bad_pix,
                                                                 loop_vars=(gm, mu_krn, dt, bad_num, percent_change, index, Niter))




    # define a near-local smoothing kernel for replacing bad pixels
    skern_size = np.array([5, 5, 5], dtype=np.int64)
    sk2 = (skern_size - 1) / 2
    smoothing_kernel = np.empty(skern_size, dtype=np.float32)
    for i in range(skern_size[0]):
        for j in range(skern_size[1]):
            for k in range(skern_size[2]):
                r = np.sqrt((i - sk2[0]) ** 2 + (j - sk2[1]) ** 2 + (k - sk2[2]) ** 2, dtype=np.float32)
                smoothing_kernel[i, j, k] = np.exp(-r) / (1 + r * r * r)

    # print(smoothing_kernel)

    smoothing_kernel = np.expand_dims(smoothing_kernel, -1)
    smoothing_kernel = np.expand_dims(smoothing_kernel, -1)
    smoothing_krn = tf.convert_to_tensor(smoothing_kernel, dtype=np.float32, name='smoothing_krn')

    # smooth data
    g_data = tf.multiply(gm, dt, name='g_data')
    n_norm = tf.nn.conv3d(gm, smoothing_krn, strides=[1, 1, 1, 1, 1], padding="SAME", name='norm')
    sg_data = tf.nn.conv3d(g_data, smoothing_krn, strides=[1, 1, 1, 1, 1], padding='SAME', name='sg_data')
    sg_data = tf.divide(sg_data, n_norm, name='sg_data')


    # Replace bad pixels
    all_bad = tf.subtract(1.0, gm, name='all_bad')
    fixed_pix = tf.multiply(sg_data, all_bad, name='fixed_pix')
    dt = tf.add(g_data, fixed_pix, name='dt')


    # Start tensorflow session
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        # initialize debugger
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Open summary writer
        tb_writer = tf.summary.FileWriter('./data')

        # Visualize graph
        tb_writer.add_graph(sess.graph)

        # Evaluate graph
        sess.run(init)
        (dt, gm, bad_num,index) = sess.run([dt, gm, bad_num, index])
        # (dt, gm, bad_num) = sess.run([dt, gm, bad_num])


        print('Number of iterations', index)  # why is this always 4?
        print('Bad Pix Found', bad_num)
        return (np.squeeze(dt), np.squeeze(gm), bad_num)
