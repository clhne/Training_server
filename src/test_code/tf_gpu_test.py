#test tf gpu

import tensorflow as tf
'''
for i in range(1,1):
    # Creates a graph.
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))

# Allowing GPU memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
'''

# Using multiple GPUs

# Creates a graph.
c = []
#for d in ['/device:GPU:0', '/device:GPU:1']:
for d in ['/device:GPU:1']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#method 1
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#sess = tf.Session(config=tf.ConfigProto().gpu_options.allow_growth=True)
sess = tf.Session(config = config)
'''
#method 2
gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True, gpu_options = gpu_options))
'''
# Runs the op.
print(sess.run(sum))
