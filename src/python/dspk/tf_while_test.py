import tensorflow as tf
import numpy as np

sess = tf.Session()
x = tf.constant(10.)
i = tf.constant(0.)

def body(x,i):
    x = tf.subtract(x,i)
    i = tf.add(i,1.37)
    return (x,i)

def cond(x,i):
    return tf.greater_equal(x,i)

why = tf.while_loop(cond,body,loop_vars=(x,i))


print(sess.run(why))