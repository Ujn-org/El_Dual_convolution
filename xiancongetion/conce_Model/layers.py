"""
--author: Chuanhao Dong;
--time:2020/11/13 14:09;
--file:layers;
--Descriptions:
     
"""
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as Fun
from conce_Model.four_dimis_TCN import DualChannelTemporalConvolution
from sklearn.metrics import confusion_matrix


def embed_conv1d(inputs, ker_size, output_channels, name):
    embed_data = layers.Conv1D(output_channels, kernel_size=ker_size, name=name)(inputs)
    return embed_data


def embed_conv2d(inputs, ker_size, output_channels, name):
    embed_data = layers.Conv2D(output_channels, kernel_size=ker_size, padding="same", name=name)(inputs)
    return embed_data


def gated_layer(hist, recent):
    sigmoid_gate = tf.sigmoid(tf.add(hist, recent[:, :, :, :64]))
    return sigmoid_gate*recent[:, :, :, -64:]


def output_layer():
    pass


def trainer(trainset_input, trainset_target,):
    rencentData, historyData, Attr = trainset_input
    target = trainset_target

    embed_Attr = embed_conv1d(Attr, 5, 64, "attr_embedding")
    historydata = embed_conv2d(tf.transpose(historyData, [0, 1, 3, 2]), 3, 64, "his_dataEmbedding")
    recentdata = embed_conv2d(rencentData, 3, 128, "ren_dataEmbedding")
    embed_Attr = tf.transpose(tf.concat([tf.expand_dims(embed_Attr, -1)]*4, axis=-1), [0, 1, 3, 2])

    hist = tf.add(historydata, embed_Attr)
    gated_data = gated_layer(hist, recentdata)
    temporal_data = DualChannelTemporalConvolution()(gated_data)

    predicts = tf.nn.softmax(temporal_data, -1)
    accuracy = tf.cast(tf.equal(tf.arg_max(predicts, -1), tf.arg_max(target, -1)), tf.int32)
    loss = tf.reduce_mean(tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=predicts, dim=0), [-1, 1]), axis=0)
    confusion_matrixs = tf.confusion_matrix(tf.arg_max(predicts, -1), tf.arg_max(target, -1), num_classes=4)
    return accuracy, loss, confusion_matrixs


def tester():
    pass
