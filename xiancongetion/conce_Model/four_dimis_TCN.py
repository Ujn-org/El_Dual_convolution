"""
--author: Chuanhao Dong;
--time:2020/11/13 20:32;
--file:four_dimis_TCN;
--Descriptions:
     
"""
import inspect
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Activation, SpatialDropout2D, Lambda
from tensorflow.python.keras.layers import Layer, BatchNormalization, LayerNormalization, Conv2D
import tensorflow as tf

###############################
# dual channel  residual block#
###############################


class DualChannelReasidualBlock(Layer):
    def __init__(self,
                 dilation_rate,
                 nb_filters,
                 kernel_size,
                 padding,
                 activation='relu',
                 dropout_rate=0.5,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 last_block=True,
                 **kwargs
                 ):
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.kernel_initializer = kernel_initializer
        self.last_block = last_block
        super(DualChannelReasidualBlock, self).__init__()
        self.branch1_layers = []
        self.branch2_layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

    def branche1_add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.branch1_layers.append(layer)
        self.branch1_layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.branch1_layers[-1].compute_output_shape(self.res_output_shape)

    def branche2_add_and_activate_layer(self, layer):
        """Helper function for building layer

        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.

        """
        self.branch2_layers.append(layer)
        self.branch2_layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.branch2_layers[-1].compute_output_shape(self.res_output_shape)

    def build(self, input_shape):
        with K.name_scope(self.name):  # name scope used to make sure weights get unique names
            self.res_output_shape = input_shape
            with K.name_scope("temporal_branch1"):
                name = 'branch1_conv2D'
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    self.branche1_add_and_activate_layer(Conv2D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))

                if self.use_batch_norm:
                    self.branche1_add_and_activate_layer(BatchNormalization())
                elif self.use_layer_norm:
                    self.branche1_add_and_activate_layer(LayerNormalization())

                self.branche1_add_and_activate_layer(Activation('linear'))
                self.branche1_add_and_activate_layer(SpatialDropout2D(rate=self.dropout_rate))
            self.res_output_shape = input_shape
            with K.name_scope("temporal_branch2"):
                name = 'branch2_conv2D'
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    self.branche2_add_and_activate_layer(Conv2D(filters=self.nb_filters,
                                                        kernel_size=self.kernel_size,
                                                        dilation_rate=self.dilation_rate,
                                                        padding=self.padding,
                                                        name=name,
                                                        kernel_initializer=self.kernel_initializer))

                if self.use_batch_norm:
                    self.branche2_add_and_activate_layer(BatchNormalization())
                elif self.use_layer_norm:
                    self.branche2_add_and_activate_layer(LayerNormalization())

                self.branche2_add_and_activate_layer(Activation('relu'))
                self.branche2_add_and_activate_layer(SpatialDropout2D(rate=self.dropout_rate))

            if self.last_block:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'conv2D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv2D(filters=self.nb_filters,
                                                   kernel_size=[1, 1],
                                                   padding='same',
                                                   name=name,
                                                   kernel_initializer=self.kernel_initializer)
            else:
                self.shape_match_conv = Lambda(lambda x: x, name='identity')

            self.shape_match_conv.build(input_shape)
            self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)

            self.final_activation = Activation(self.activation)
            self.final_activation.build(self.res_output_shape)  # probably isn't necessary

            # this is done to force Keras to add the layers in the list to self._layers
            for layer in self.branch1_layers:
                self.__setattr__(layer.name, layer)
            for layer in self.branch2_layers:
                self.__setattr__(layer.name, layer)
            super(DualChannelReasidualBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        branch1_x = branch2_x = inputs
        self.layers_outputs = [branch1_x]

        for layer in self.branch1_layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            branch1_x = layer(branch1_x, training=training) if training_flag else layer(branch1_x)
            self.layers_outputs.append(branch1_x)

        for layer in self.branch2_layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            branch2_x = layer(branch2_x, training=training) if training_flag else layer(branch2_x)
            self.layers_outputs.append(branch2_x)

        branch_x = Activation("sigmoid")(branch1_x + branch2_x)
        x2 = self.shape_match_conv(inputs)
        self.layers_outputs.append(x2)

        res_x = layers.add([x2, branch_x])
        self.layers_outputs.append(res_x)
        res_act_x = self.final_activation(res_x)

        self.layers_outputs.append(res_act_x)

        return [res_act_x, branch_x]


class DualChannelTemporalConvolution(Layer):
    def __init__(self,
                 dilations=(1, 2, 4, 8, 16, 32),
                 nb_filters=64,
                 kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 stack=1,
                 ):
        self.dilations = dilations
        self.nb_stacks = stack
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation = activation
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        super(DualChannelTemporalConvolution, self).__init__(self)

    def build(self, input_shape):
        self.build_output_shape = input_shape
        for i in range(self.nb_stacks):
            for dilat in self.dilations:
                self.residual_blocks.append(DualChannelReasidualBlock(dilat, self.nb_filters, self.kernel_size, self.padding,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)
        super(DualChannelTemporalConvolution, self).__init__(self)

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.residual_blocks:
            x, skip_out = layer(x)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append([x, skip_out])
        squess_out = []
        for layer_out in self.skip_connections:
            squess_out.append(K.expand_dims(layer_out))
        concat_out = K.concatenate(squess_out)
        x = layers.Conv3D(4, [5, 4, 64], padding="valid", activation="relu")(concat_out)
        x = K.reshape(x, [-1, 4])
        return x

# inputs = tf.placeholder(dtype=tf.float32, shape=[64, 5, 4, 64])
# dual_channel_Tcn = DualChannelTemporalConvolution()(inputs)
