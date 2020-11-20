"""
--author: Chuanhao Dong;
--time:2020/11/13 20:34;
--file:test_keras;
--Descriptions:
     
"""

import tensorflow as tf
from tensorflow.python.keras import layers, Model, Input
from tensorflow.python.keras import backend
# input_data = tf.placeholder(dtype=tf.float32, shape=[64, 5, 4, 4])
input_data = Input([5, 4, 4], 64)

output = layers.Conv2D(64, [3, 3])(input_data)

model = Model(inputs=[input_data], outputs=output)
model