# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

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
    
#logging device placement
a = tf.constant([1.0,2.0, 3.0, 4.0, 5.0, 6.0],
shape=[2,3],name='a')
b = tf.constant([1.0,2.0, 3.0, 4.0, 5.0, 6.0],
shape=[3,2],name='b')
c = tf.matmul(a,b)
#creates a session with log_device_placement 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#run the op.
print(sess.run(c))