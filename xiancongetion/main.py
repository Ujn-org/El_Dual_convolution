"""
--author: Chuanhao Dong;
--time:2020/11/13 14:14;
--file:main;
--Descriptions:
     
"""
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
from util.fun import plot_confusion_matrix
from conce_Model.layers import trainer,tester
from util.data_load import loadDataset
import argparse
import numpy as np

args = argparse.ArgumentParser()
args.add_argument("--data_file", default={"train_data": "./data/20190701.txt", "topo": "./data/topo.txt", "attr": "./data/attr.txt"}, help="files of train data")
args.add_argument("--batch_size", default=128, help="batch size")
args.add_argument("--epochs", default=64, help="batch size")
args.add_argument("--learning_rate", default=1e-3, help="batch size")
args.add_argument("--net_config", default="./config.txt", help="the details of network")

arg_seq = args.parse_args()


Temporal_data = loadDataset(arg_seq.data_file["train_data"], arg_seq.data_file["topo"], arg_seq.data_file["attr"], arg_seq.batch_size)([0.8, 0.9])
train_data = Temporal_data.load_trainData()
# topo = keras.Input([9], arg_seq.batch_size)
rencent_data = tf.placeholder(dtype=tf.float32, shape=[None, 5, 4, 1])
past_data = tf.placeholder(dtype=tf.float32, shape=[None, 5, 4, 4])
attr = tf.placeholder(dtype=tf.float32, shape=[None, 9, 1])
target = tf.placeholder(dtype=tf.int64, shape=[None, 1])
targetinput = tf.reshape(tf.one_hot(target, depth=4), [-1, 4])


if __name__ == "__main__":
    accuracy, loss, confusemetrixs = trainer([rencent_data, past_data, attr], targetinput)
    train_varibles = tf.trainable_variables()
    opt = tf.train.GradientDescentOptimizer(learning_rate=arg_seq.learning_rate)
    trainOP = opt.minimize(loss)
    initer = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(initer)
        print("model training.......")
        for ep in range(0, arg_seq.epochs):
            for data in train_data:
                loss_data, accuracy_data, confusemetrixs_data = sess.run([loss, accuracy, confusemetrixs], feed_dict={
                    rencent_data: np.reshape(data["recent_data"], [-1, 5, 4, 1]),
                    past_data: data["past_data"],
                    attr: np.reshape(data["attr"], [-1, 9, 1]),
                    target: np.reshape(data["target"], [-1, 1]),
                })
                confuseima = plot_confusion_matrix(confusemetrixs_data, "confuse metrixs_label", "confuse metrixs")
                confuseima.show()
                print(f"train epoch{ep},loss:{loss_data},accurcy:{np.divide(np.sum(accuracy_data),arg_seq.batch_size) * 100}%! \n")
                sess.run(trainOP, feed_dict={
                    rencent_data: np.reshape(data["recent_data"], [-1, 5, 4, 1]),
                    past_data: data["past_data"],
                    attr: np.reshape(data["attr"], [-1, 9, 1]),
                    target: np.reshape(data["target"], [-1, 1]),
                })
