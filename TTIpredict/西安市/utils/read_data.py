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
        self.temporal_or_data[2], self.tti_mean, self.tti_std = self.Z_score(self.temporal_or_data[2]*10)
        self.temporal_or_data[3], self.speed_mean, self.speed_std = self.Z_score(self.temporal_or_data[3])
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
            if nodes > 3000:
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
        print(f"road_id: {road_id},uper_id:{uper_id}, downer_id:{downer_id}")

        temporal_data = self.temporal_or_data[self.temporal_or_data[0] == road_id].to_numpy()
        for i in range(len(temporal_data) - self.h_step - self.pre_step):
            single_data = {}
            single_data["temporal_data"] = np.array(temporal_data[i:i + self.h_step + self.pre_step,  (2, 3)])
            single_data["uper_data"] = np.array([self.temporal_or_data[self.temporal_or_data[0] == int(uper_id[0])].to_numpy()[
                                       i:i + self.h_step, (2, 3)], self.temporal_or_data[self.temporal_or_data[0] == int(uper_id[1])].to_numpy()[
                                       i:i + self.h_step, (2, 3)]])
            single_data["downer_data"] = np.array([self.temporal_or_data[self.temporal_or_data[0] == int(downer_id[0])].to_numpy()[
                                         i:i + self.h_step, (2, 3)], self.temporal_or_data[self.temporal_or_data[0] == int(downer_id[1])].to_numpy()[
                                         i:i + self.h_step, (2, 3)]])
            self.temporaldata.append(single_data)

    def loadTemporaldata(self, restore):
        if restore:
            self.loadTopo()
            print(">>load temporal data ... ... ...<<")
            for node in self.Topo:
                uper1_len = self.temporal_or_data[self.temporal_or_data[0] == int(str(node["uper_id"]).split(",")[0])].__len__()
                uper2_len = self.temporal_or_data[self.temporal_or_data[0] == int(str(node["uper_id"]).split(",")[1])].__len__()
                dower1_len = self.temporal_or_data[self.temporal_or_data[0] == int(str(node["downer_id"]).split(",")[0])].__len__()
                dower2_len = self.temporal_or_data[self.temporal_or_data[0] == int(str(node["downer_id"]).split(",")[1])].__len__()
                node_len = self.temporal_or_data[self.temporal_or_data[0] == int(node["id"])].__len__()
                if uper1_len==uper2_len==dower1_len==dower2_len==node_len:
                    self.loadPointData(node["id"], str(node["uper_id"]).split(","), str(node["downer_id"]).split(","))
            random_index = np.arange(len(self.temporaldata))
            np.random.shuffle(random_index)
            self.temporaldata = np.array(self.temporaldata)[random_index]
            temptoal_all = np.array([np.array(ele["temporal_data"]) for ele in self.temporaldata])
            uper_all = np.array([np.array(ele["uper_data"]) for ele in self.temporaldata])
            downer_all = np.array([np.array(ele["downer_data"]) for ele in self.temporaldata])
            np.save("./temporal_all.npy", temptoal_all)
            np.save("./uper_all.npy", uper_all)
            np.save("./downer_all.npy", downer_all)
        else:
            temptoal_all =  np.load("temporal_all_noscale.npy")
            uper_all = np.load("uper_all_noscale.npy")
            downer_all = np.load("downer_all_noscale.npy")
        return temptoal_all, uper_all, downer_all

    def Z_score(self, Data):
        mean = Data.to_numpy().mean(dtype=np.float32)
        std = Data.to_numpy().std(dtype=np.float32)
        return (Data-mean)/std, mean, std

    def Re_Z_score(self, Data, mean, std):
        return  Data*std+mean
