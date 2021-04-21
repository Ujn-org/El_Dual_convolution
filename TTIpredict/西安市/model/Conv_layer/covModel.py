import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Conv2D, Conv1D, Dense
from tensorflow.python.keras import backend as F
import numpy as np


class BaseModel(Layer):
    def __init__(self, batch_size=64, units=(32, 64, 128), kernel_size=(1, 3, 5)):
        """
        :param batch_size: 64
        :param units:  default:
        :param kernel_size:
        """
        self.batch_size = batch_size
        self.units = units
        self.kernel_size = kernel_size
        super(BaseModel).__init__()

    def BaseModel(self, temporal_data, uper_data, downer_data):
        """

        :param temporal_data: shape[-1,1,12, 2]
        :param uper_data: shape[-1,2,12,2]
        :param downer_data: shape[-1,2,12,2]
        :return: shape[-1, 5, 2]
        """
        con_temporal_outputs = Conv2D(self.units[2], self.kernel_size[0],padding="same", activation="tanh")(temporal_data)
        con_uper_outputs = Conv2D(self.units[1], self.kernel_size[0], padding="same", activation="tanh")(uper_data)
        con_downer_outputs = Conv2D(self.units[1], self.kernel_size[0], padding="same", activation="tanh")(downer_data)

        uper_temp = F.concatenate([con_temporal_outputs[:, :, :, 0:int(self.units[2]/2)], con_uper_outputs], axis=1)
        down_temp = F.concatenate([con_temporal_outputs[:, :, :, int(self.units[2]/2):self.units[2]], con_downer_outputs], axis=1)

        squUper = Conv2D(self.units[0], self.kernel_size[1], padding="valid", activation="relu")(uper_temp)
        squDowner = Conv2D(self.units[0], self.kernel_size[1], padding="valid", activation="relu")(down_temp)

        concat_features = F.concatenate([F.squeeze(squUper, 1), F.squeeze(squDowner, 1)], axis=-1)
        model_out = Conv1D(2, self.kernel_size[1],activation="relu")(concat_features)

        model_tti_out = Conv1D(1, self.kernel_size[1])(model_out[:, :, 0:1])
        model_speed_out = Conv1D(1, self.kernel_size[1])(model_out[:, :, 1:2])
        return model_tti_out, model_speed_out