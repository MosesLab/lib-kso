
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

sess = tf.Session()

kernel = gauss(10,10,5,5)
image = mpimg.imread('../../../data/sun.jpg')
plt.figure()
plt.imshow(kernel)
plt.figure()
plt.imshow(image)

kernel = np.expand_dims(kernel,-1)
kernel = np.expand_dims(kernel,-1)
kernel = np.tile(kernel, [1,1,3,3])

image = np.expand_dims(image,0)


krn = tf.convert_to_tensor(kernel, dtype=np.float32)
img = tf.convert_to_tensor(image, dtype=np.float32)

strides = [1,1,1,1]

new_img = tf.nn.conv2d(img, krn, strides, padding="SAME")

with sess.as_default():
    new_image = new_img.eval()

print(new_image)

new_image = np.squeeze(new_image)

new_image = (new_image - np.min(new_image)) / (np.max(new_image) - np.min(new_image))

plt.figure()
plt.imshow(new_image)

plt.show()


