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
# from util.fun import plot_confusion_matrix
from conce_Model.layers import basemodel, tester
from util.data_load import loadDataset
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score

args = argparse.ArgumentParser()
args.add_argument("--data_file", default={"train_data": "./data/balance.txt", "val_data": "./data/valid.txt", "test_data": "./data/testdata.txt", "topo": "./data/topo.txt", "attr": "./data/attr.txt"}, help="files of train data")
args.add_argument("--batch_size", default=128, help="batch size")
args.add_argument("--epochs", default=10, help="batch size")
args.add_argument("--learning_rate", default=1e-5, help="batch size")
args.add_argument("--net_config", default="./config.txt", help="the details of network")
args.add_argument("--training", default=True, help="the details of network")

arg_seq = args.parse_args()


Temporal_data = loadDataset(arg_seq.data_file["train_data"], arg_seq.data_file["topo"], arg_seq.data_file["attr"], arg_seq.batch_size)([1.0, 1.0])
Temporal_val_data = loadDataset(arg_seq.data_file["val_data"], arg_seq.data_file["topo"], arg_seq.data_file["attr"], arg_seq.batch_size)([1.0, 1.0])
Temporal_test_data = loadDataset(arg_seq.data_file["test_data"], arg_seq.data_file["topo"], arg_seq.data_file["attr"], arg_seq.batch_size)([1.0, 1.0])

# topo = keras.Input([9], arg_seq.batch_size)
rencent_data = tf.placeholder(dtype=tf.float32, shape=[None, 5, 4, 1])
past_data = tf.placeholder(dtype=tf.float32, shape=[None, 5, 4, 4])
attr = tf.placeholder(dtype=tf.float32, shape=[None, 9, 1])
target = tf.placeholder(dtype=tf.int64, shape=[None, 1])
targetinput = tf.reshape(tf.one_hot(target-1, depth=3), [-1, 3])


def trainer():
    accuracy = tf.cast(tf.equal(tf.arg_max(predicts, -1), tf.arg_max(targetinput, -1)), tf.int32)
    loss = tf.reduce_mean(tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels=targetinput, logits=predicts, dim=1), [-1, 1]), axis=0)
    confusion_matrixs = tf.confusion_matrix(tf.arg_max(predicts, -1), tf.arg_max(targetinput, -1), num_classes=3)
    return accuracy, loss, confusion_matrixs


if __name__ == "__main__":
    predicts = basemodel([rencent_data, past_data, attr])
    accuracy, loss, confusion_matrixs = trainer()
    train_varibles = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(learning_rate=arg_seq.learning_rate)
    trainOP = opt.minimize(loss)
    initer = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(initer)
        if arg_seq.training:
            print("model training.......")
            for ep in range(0, arg_seq.epochs):
                train_data = Temporal_data.load_trainData()
                for data in train_data:
                    loss_data, accuracy_data, predict_data, confusemetrixs_data, targetinput_data = sess.run([loss, accuracy, predicts, confusion_matrixs, targetinput], feed_dict={
                        rencent_data: np.reshape(data["recent_data"], [-1, 5, 4, 1]),
                        past_data: data["past_data"],
                        attr: np.reshape(data["attr"], [-1, 9, 1]),
                        target: np.reshape(data["target"], [-1, 1]),
                    })
                    # confuseima = plot_confusion_matrix(confusemetrixs_data, "confuse metrixs_label", "confuse metrixs")
                    # confuseima.show()
                    print(f"train epoch{ep},loss:{loss_data},predicts:{predict_data},accurcy:{np.divide(np.sum(accuracy_data),128)*100}%! \n")
                    sess.run(trainOP, feed_dict={
                         rencent_data: np.reshape(data["recent_data"], [-1, 5, 4, 1]),
                        past_data: data["past_data"],
                        attr: np.reshape(data["attr"], [-1, 9, 1]),
                        target: np.reshape(data["target"], [-1, 1]),
                    })
                if ep % 2 == 0:
                    saver.save(sess, "./output/model.ckpt")
                val_data = Temporal_val_data.load_valData()
                for data in val_data:
                    loss_data, accuracy_data, predict_data, confusemetrixs_data, targetinput_data = sess.run(
                        [loss, accuracy, predicts, confusion_matrixs, targetinput], feed_dict={
                            rencent_data: np.reshape(data["recent_data"], [-1, 5, 4, 1]),
                            past_data: data["past_data"],
                            attr: np.reshape(data["attr"], [-1, 9, 1]),
                            target: np.reshape(data["target"], [-1, 1]),
                        })
                    print(f"val_data: loss:{loss_data}, accuracy:{accuracy_data}!\n")
                    f1_scoredata_list = f1_score(np.argmax(targetinput_data, -1)+1, np.argmax(predict_data, -1)+1, labels=[1, 2, 3], average=None)
                    print(f"f1_score_list:{f1_scoredata_list},power_f1_score:{0.2*f1_scoredata_list[0]+0.2*f1_scoredata_list[1]+0.6*f1_scoredata_list[2]}.\n")
        else:
            latestckpt = tf.train.latest_checkpoint("./output/")
            saver.restore(sess, latestckpt)
            train_data = Temporal_test_data.load_trainData()
            test_labels = []
            for i, data in enumerate(train_data):
                predict_data = sess.run(
                    tf.arg_max(predicts, dimension=-1), feed_dict={
                        rencent_data: np.reshape(data["recent_data"], [-1, 5, 4, 1]),
                        past_data: data["past_data"],
                        attr: np.reshape(data["attr"], [-1, 9, 1]),
                        target: np.reshape(data["target"], [-1, 1]),
                    })
                print(f"{i}th batch data!\n")
                test_predict_data = predict_data
                test_labels.append(test_predict_data+1)
            pd.DataFrame(test_labels).to_csv("test_label.txt")
