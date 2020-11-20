import pandas as pd
import numpy as np

####################
    # timestep_data_list  saved the recent data of speed states and the
    # past_tiemstep_data_list saved the past weeks data of speed states
    # we use the two blocks to input the rencent data and the past data.
    # A gated mechine is employed to add the attributes information into the network.
####################


def Normalation_data(data):
    batch_time_id_list, batch_timestep_data_list = [], []
    for item in data:
        timestep_list = str(item).split(" ")
        time_id_list, timestep_data_list = [], []
        for elem_data in timestep_list:
            time_id, timestep_data = elem_data.split(":")
            timestep_data = timestep_data.split(",")
            time_id_list.append(time_id)
            timestep_data_list.append(timestep_data)
        batch_time_id_list.append(time_id_list), batch_timestep_data_list.append(timestep_data_list)
    batch_time_id_list = np.array(batch_time_id_list, dtype=np.float32)
    batch_timestep_data_list = np.array(batch_timestep_data_list, dtype=np.float32)
    mean, std = np.mean(batch_timestep_data_list, axis=0).mean(axis=0), np.std(batch_timestep_data_list, axis=0).mean(axis=0)
    batch_timestep_data_list_normal = (batch_timestep_data_list - mean)/std
    return batch_timestep_data_list_normal


class Dataset():
    def __init__(self, train_data, val_data, test_data, attr_data, batch_size):
        self.trainset = train_data
        self.testset = test_data
        self.valset = val_data
        self.attr_data = attr_data
        self.batch_size = batch_size

    def load_trainData(self):
        for i in range(0, len(self.trainset), self.batch_size):
            batch_data = self.trainset[i: i+self.batch_size]
            yield self.split_infor(batch_data)

    def load_valData(self):
        for i in range(0, len(self.valset), self.batch_size):
            batch_data = self.valset[i: i+self.batch_size]
            yield self.split_infor(batch_data)

    def load_testData(self):
        for i in range(0, len(self.testset), self.batch_size):
            batch_data = self.testset[i: i+self.batch_size]
            yield self.split_infor(batch_data)

    def split_infor(self, oringindata):
        id_list = [np.float64(ids) for ids in oringindata[0].to_list()]
        y_target = oringindata[1].to_numpy(dtype=np.int)   # target
        timestep_data_list = Normalation_data(oringindata[4])  # recent data
        past_timestep_data_list = []
        for j in range(5, 9):
            temp_timestep_data_list = Normalation_data(oringindata[j])
            temp_timestep_data_list = np.expand_dims(np.array(temp_timestep_data_list), axis=-1)
            past_timestep_data_list.append(temp_timestep_data_list)
        past_timestep_data = np.concatenate(past_timestep_data_list, axis=-1)  # past data

        batch_attr = []
        for id_element in id_list:
            temp_data = self.attr_data[self.attr_data[0] == id_element].to_numpy()[:]
            batch_attr.append(temp_data)
        batch_attr = np.concatenate(batch_attr, axis=0)  # attribute data
        return {"target": y_target, "recent_data": timestep_data_list, "past_data": past_timestep_data, "attr": batch_attr}


class loadDataset():
    def __init__(self, data_file, topo_file, attr_file, batch_size):
        self.datafile = data_file
        self.topofile = topo_file
        self.attr_file = attr_file
        self.batch_size = batch_size
        self.topo = self.loadTopo()
        self.attr = self.loaAttr()

    def __call__(self, rate, *args, **kwargs):
        return self.loadData(rate)

    def loadData(self, rate):
        self.Topo_data = self.loadTopo()
        self.origin_data = pd.read_csv(self.datafile, ";", header=None)
        data = self.origin_data[0].str.split(" ", expand=True)
        data_combine = pd.concat([data[0], data[1], data[2], data[3], self.origin_data[[1, 2, 3, 4, 5]]], axis=1).to_numpy()
        data_combine = pd.DataFrame(data_combine)
        train_data, val_data, test_data = data_combine[:int(len(data_combine)*rate[0])],\
                                           data_combine[int(len(data_combine)*rate[0]):int(len(data_combine)*rate[1])],\
                                           data_combine[int(len(data_combine) * rate[1]):]
        return Dataset(train_data, val_data, test_data, self.attr, self.batch_size)

    def loadTopo(self):
        orderlist = pd.read_csv(self.topofile, "\t", header=None)
        return orderlist

    def loaAttr(self):
        attr = pd.read_csv(self.attr_file, "\t", header=None).to_numpy()
        mean_list = np.mean(attr[:, 1:], 0)
        std_list = np.std(attr[:, 1:], 0)
        attr_data = np.divide(np.subtract(attr[:, 1:], mean_list), std_list)
        attr[:, 1:] = attr_data
        attr[:, 0] = [round(i, 0) for i in attr[:, 0]]
        return pd.DataFrame(attr)


# Data = loadDataset("../data/traffic/traffic/20190701.txt", "../data/topo.txt", "../data/attr.txt", 64)
# temporal_data = Data.loadData([0.8, 0.9])
# trainData = temporal_data.load_trainData()
