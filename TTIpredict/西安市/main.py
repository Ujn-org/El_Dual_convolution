import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Conv2D, Conv1D
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras import backend as F
import numpy as np
import argparse
import pandas as pd
from utils.read_data import dataLoad
from model.Conv_layer.covModel import BaseModel
from model.Dual_channel_conv import Dual_channel_Conv
from utils.factor import MAE, MAPE, RMSE


parser = argparse.ArgumentParser()
parser.add_argument("--temporal_file", default="./road/西安市_first_episoid.txt",help="temporal file path")
parser.add_argument("--Topo_file", default="./boundary.txt",help="topo file path")
parser.add_argument("--epoch", default=1000, help="training epochs.")
parser.add_argument("--batch_size", default=128, help=" batch size.")
parser.add_argument("--learning_rate", default=0.0001, help=" batch size.")
parser.add_argument("--historical_steps", default=12, help="h time step of historical information.")
parser.add_argument("--pre_steps", default=6, help=" p time step in the future will be predicted.")
parser.add_argument("--units", default=[32, 64, 128], help=" batch size.")
args = parser.parse_args()
print(args)
print("<==================Loading data, maybe this will take a period time ! .... ==============>")
data_loader = dataLoad(args.temporal_file, args.Topo_file, args.historical_steps,  args.pre_steps, args.batch_size)
data_iter = data_loader.loadTemporaldata(True)

temporal_input = Input([1, args.historical_steps, 2], args.batch_size, "temporal_input")
uper_input = Input([2, args.historical_steps, 2], args.batch_size, "uper_input")
downer_input = Input([2, args.historical_steps, 2], args.batch_size, "downer_input")
labels = Input([args.pre_steps, 2], args.batch_size, "labels")

conv_model = BaseModel()
model_output_tti, model_output_speed = conv_model.BaseModel(temporal_input, uper_input, downer_input)

resc_output_tti = data_loader.Re_Z_score(model_output_tti, data_loader.tti_mean, data_loader.tti_std)
resc_output_speed = data_loader.Re_Z_score(model_output_speed, data_loader.speed_mean, data_loader.speed_std)
resc_lable_tti = data_loader.Re_Z_score(labels[:, :, 0:1], data_loader.tti_mean, data_loader.tti_std)
resc_lable_speed = data_loader.Re_Z_score(labels[:, :, 1:2], data_loader.speed_mean, data_loader.speed_std)

ttiLoss = tf.nn.l2_loss(resc_output_tti - resc_lable_tti)
speedLoss = tf.nn.l2_loss(resc_output_speed - resc_lable_speed)
apha = F.variable(1, dtype=tf.float32)
lamda = F.variable(1, dtype=tf.float32)
global_steps = tf.Variable(0, trainable=False)
epoch_step = int(len(data_iter[0])*0.8)/args.batch_size
loss = tf.exp(apha)/(tf.exp(apha)+tf.exp(lamda)) * ttiLoss + tf.exp(lamda)/(tf.exp(apha)+tf.exp(lamda))*speedLoss
lr = tf.train.exponential_decay(args.learning_rate, global_steps, decay_steps=50 * epoch_step, decay_rate=0.6, staircase=True)
opt_handel = tf.train.RMSPropOptimizer(args.learning_rate)
opter = opt_handel.minimize(loss)


if __name__ == "__main__":
    initer = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initer)
        print( "<======Successfully  loading data, Model is building .... =========>")
        for i in range(args.epoch):
            for j in (range(0, int(len(data_iter[0])*0.8)-args.batch_size, args.batch_size)):
                temporal = data_iter[0][j:j+args.batch_size]
                temporal_histor = np.expand_dims(temporal[:, 0:args.historical_steps, :], axis=1)
                temporal_pre = np.array(temporal[:, args.historical_steps:args.historical_steps+args.pre_steps, :])
                uper = np.array(data_iter[1][j:j+args.batch_size])
                downer = np.array(data_iter[2][j:j+args.batch_size])
                dic = {temporal_input: temporal_histor, uper_input: uper, downer_input: downer, labels: temporal_pre}
                tti_outputsData, speed_outputsData, tti_labelData, speed_labelData, lossData,_ = sess.run([resc_output_tti, resc_output_speed, resc_lable_tti, resc_lable_speed,loss, opter], feed_dict=dic)
            for j in (range(int(len(data_iter[0])*0.8)-args.batch_size, len(data_iter[0]) - args.batch_size, args.batch_size)):
                temporal = data_iter[0][j:j + args.batch_size]
                temporal_histor = np.expand_dims(temporal[:, 0:args.historical_steps, :], axis=1)
                temporal_pre = np.array(temporal[:, args.historical_steps:args.historical_steps + args.pre_steps, :])
                uper = np.array(data_iter[1][j:j + args.batch_size])
                downer = np.array(data_iter[2][j:j + args.batch_size])
                dic = {temporal_input: temporal_histor, uper_input: uper, downer_input: downer, labels: temporal_pre}
                tti_outputsData, speed_outputsData, tti_labelData, speed_labelData, lossData = sess.run([resc_output_tti, resc_output_speed, resc_lable_tti, resc_lable_speed, loss], feed_dict=dic)
                ttiMae, speedMae = MAE(tti_outputsData, tti_labelData), MAE(speed_outputsData,speed_labelData)
                ttiMape, speedMape = MAPE(tti_outputsData,tti_labelData), MAPE(speed_outputsData,speed_labelData)
                ttiRmse, speedRmse = RMSE(tti_outputsData,tti_labelData), RMSE(speed_outputsData,speed_labelData)
                print(f"the loss of epoch- {i}th is {lossData:.3f} \nttiMAE :{ttiMae:.3f}. ttiMAPE:{ttiMape:.3f}. ttiRMSE:{ttiRmse:.3f}!")
                print(f"speedMAE :{speedMae:.3f}. speedMAPE:{speedMape:.3f}. speedRMSE:{speedRmse:.3f}!\n")