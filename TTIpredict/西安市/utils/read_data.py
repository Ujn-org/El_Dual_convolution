import pandas as pd
import numpy as np
import json


class dataLoad():
    def __init__(self, temporal_file, spatial_file, h_step, pre_step, bathsize):
        """
        functions:
            this class is used to load the original data set and return the data can be input to the model.
        :param temporal_file: temporal data file
        :param spatial_file: information of topo
        :param h_step:  h time step of historical inforamtion
        :param pre_step:  h time step in the future will be predicted
        :param bathsize:  batchsize of minibatch
        """
        self.temporal_file = temporal_file
        self.spatial_file = spatial_file
        self.temporal_or_data = pd.read_csv(self.temporal_file, header=None)
        self.spatial_or_data = pd.read_csv(self.spatial_file, "\t")
        self.h_step = h_step
        self.pre_step = pre_step
        self.batchsize = bathsize
        adj_info = self.spatial_or_data["obj_name"].apply(lambda x: [str(x).split(":")[0], str(x).split(":")[1].split(",")[0],
                                                                 str(x).split(":")[1].split(",")[1]])
        adj_info = np.array([np.array(i) for i in adj_info.to_numpy()])
        adj_info = pd.DataFrame(adj_info)
        self.spatial_or_data["obj_name"] = adj_info[0]
        self.spatial_or_data = pd.concat([self.spatial_or_data, adj_info[1], adj_info[2]], axis=1)
        self.spatial_or_data.columns = ["id", "name", "geo", "uper", "downer"]
        self.temporaldata = []
        self.Topo = []


    def loadTopo(self):
        print(">>>Load Topo ... ... ... <<<")
        nodes = 0
        nodes_topo = []
        for road_id in self.spatial_or_data["id"]:
            if nodes > 2:
                break
            # road_name = self.spatial_or_data[self.spatial_or_data["id"] == road_id]["name"].to_numpy()[0]
            up_road = self.spatial_or_data[self.spatial_or_data["id"] == road_id]["uper"].to_numpy()[0]
            down_road = self.spatial_or_data[self.spatial_or_data["id"] == road_id]["downer"].to_numpy()[0]
            up_road_data = self.spatial_or_data[self.spatial_or_data["name"] == up_road]
            uper_id = up_road_data["id"]
            down_road_data = self.spatial_or_data[self.spatial_or_data["name"] == down_road]
            downer_id = down_road_data["id"]
            if len(uper_id) & len(downer_id):
                nodes_topo.append({"id": road_id, "uper_id": str(uper_id.to_numpy()[0])+","+str(uper_id.to_numpy()[1]), "downer_id": str(downer_id.to_numpy()[0])+","+str(downer_id.to_numpy()[1])})
                nodes += 1
        self.Topo = nodes_topo

    def loadPointData(self, road_id, uper_id, downer_id):
        single_data = {}
        temporal_data = self.temporal_or_data[self.temporal_or_data[0] == road_id].to_numpy()
        for i in range(len(temporal_data) - self.h_step - self.pre_step):
            single_data["temporal_data"] = temporal_data[i:i + self.h_step + self.pre_step,  (2, 3)]
            single_data["uper_data"] = [self.temporal_or_data[self.temporal_or_data[0] == int(uper_id[0])].to_numpy()[
                                       i:i + self.h_step, (2, 3)], self.temporal_or_data[self.temporal_or_data[0] == int(uper_id[1])].to_numpy()[
                                       i:i + self.h_step, (2, 3)]]
            single_data["downer_data"] = [self.temporal_or_data[self.temporal_or_data[0] == int(downer_id[0])].to_numpy()[
                                         i:i + self.h_step, (2, 3)], self.temporal_or_data[self.temporal_or_data[0] == int(downer_id[1])].to_numpy()[
                                         i:i + self.h_step, (2, 3)]]
            self.temporaldata.append(single_data)

    def loadTemporaldata(self):
        self.loadTopo()
        print(">>load temporal data ... ... ...<<")
        for node in self.Topo:
            self.loadPointData(node["id"], str(node["uper_id"]).split(","), str(node["downer_id"]).split(","))
        random_index = np.arange(len(self.temporaldata))
        np.random.shuffle(random_index)
        self.temporaldata = np.array(self.temporaldata)[random_index]
        for i in range(0, len(self.temporaldata), self.batchsize):
            yield np.array([ele["temporal_data"] for ele in self.temporaldata[i:i+self.batchsize]]), np.array([ele["uper_data"] for ele in self.temporaldata[i:i+self.batchsize]]), np.array([ele["downer_data"] for ele in self.temporaldata[i:i+self.batchsize]])

    def Z_score(self, Data):
        mean = pd.DataFrame(Data).mean
        std = pd.DataFrame(Data).std
        return (pd.DataFrame(Data)-mean)/std, mean, std

    def Re_Z_score(self, Data, mean, std):
        return  Data*std+mean
