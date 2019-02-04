import tensorflow as tf
import sys


def test():
  with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

  c = tf.matmul(a, b)
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

try:
  test()
except Exception as e:
  print("error using tensorflow cpu:\n", e)


