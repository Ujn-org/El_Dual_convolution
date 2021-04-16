import pandas as pd
import numpy as np


class dataLoad():
    def __init__(self, temporal_file, spatial_file):
        self.temporal_file = temporal_file
        self.spatial_file = spatial_file
        self.temporal_or_data = pd.read_csv(self.temporal_file, header=None)
        self.spatial_or_data = pd.read_csv(self.spatial_file, "\t")
        self.spatial_or_data["obj_name"] = self.spatial_or_data["obj_name"].apply(lambda x: [str(x).split(":")[0], str(x).split(":")[1].split(",")[0],
                                                                 str(x).split(":")[1].split(",")[1]])

    def load(self):
        for road_id in self.spatial_or_data["obj_id"]:
            self.temporal_or_data[0]


data_loader = dataLoad("./road/西安市.txt", "./boundary.txt")
pass