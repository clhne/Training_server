# Any results you write to the current directory are saved as output.
import tensorflow as tf
print(tf.__version__)
#gpu devices name, /device:GPU:0
print(tf.test.gpu_device_name())
#is built with cuda, should return Ture.
print(tf.test.is_built_with_cuda())
#is gpu available, should return True.
print(tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None))
#python -c "import tensorflow as tf; print(tf.contrib.eager.num_gpus())"   
#logging device placement
a = tf.constant([1.0,2.0, 3.0, 4.0, 5.0, 6.0],
shape=[2,3],name='a')
b = tf.constant([1.0,2.0, 3.0, 4.0, 5.0, 6.0],
shape=[3,2],name='b')
c = tf.matmul(a,b)
#creates a session with log_device_placement 
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto.gpu_options.allow_growth=True)
#run the op.
print(sess.run(c))