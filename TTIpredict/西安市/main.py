import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Conv2D, Conv1D
from tensorflow.python.keras import Input, Model
import numpy as np
import argparse

from utils.read_data import dataLoad
from model.Conv_layer.covModel import BaseModel
from model.Dual_channel_conv import Dual_channel_Conv

parser = argparse.ArgumentParser()
parser.add_argument("--temporal_file", default="./road/西安市_first_episoid.txt",help="temporal file path")
parser.add_argument("--Topo_file", default="./boundary.txt",help="topo file path")
parser.add_argument("--epoch", default=1, help="training epochs.")
parser.add_argument("--batchsize", default=64, help=" batch size.")
parser.add_argument("--historical_steps", default=12, help="h time step of historical information.")
parser.add_argument("--pre_steps", default=6, help=" p time step in the future will be predicted.")
parser.add_argument("--units", default=[32, 64, 128], help=" batch size.")
args = parser.parse_args()
print(args)
data_loader = dataLoad(args.temporal_file, args.Topo_file, args.historical_steps,  args.pre_steps, args.batchsize)
data_iter = data_loader.loadTemporaldata()
print("<==================Model is building .... ==============>")

temporal_input = Input([1, args.historical_steps, 2], args.batchsize, "temporal_input")
uper_input = Input([2, args.historical_steps, 2], args.batchsize, "uper_input")
downer_input = Input([2, args.historical_steps, 2], args.batchsize, "downer_input")
labels = Input([args.pre_steps, 2], args.batchsize, "labels")
conv_model = BaseModel()
model_output_tti, model_output_pre = conv_model.BaseModel(temporal_input, uper_input, downer_input)


if __name__ == "__main__":
    initer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initer)
        print( "<======Successfully build the Model and Loading data, maybe this will take a period time !=========>")
        for temporal, uper, downer in data_iter:
            outputs = sess.run(model_output, feed_dict={temporal_input:np.reshape(temporal[:, 0:args.historical_steps, :], [-1, 1, args.historical_steps, 2])})

