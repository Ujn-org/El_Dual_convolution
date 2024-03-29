import inspect
from typing import List

from tensorflow.python.keras import backend as K, Model, Input, optimizers
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Activation, SpatialDropout1D, Lambda, LSTM
from tensorflow.python.keras.layers import Layer, Conv1D, Dense, BatchNormalization, LayerNormalization


def is_power_of_two(num):
    return num != 0 and ((num & (num - 1)) == 0)


def adjust_dilations(dilations):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations


class ResidualBlock(Layer):

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
                 **kwargs):

        # type: (int, int, int, str, str, float, str, bool, bool, bool, dict) -> None
        """Defines the residual block for the WaveNet TCN

        Args:
            x: The previous layer in the model
            training: boolean indicating whether the layer should behave in training mode or in inference mode
            dilation_rate: The dilation power of 2 we are using for this residual block
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            padding: The padding used in the convolutional layers, 'same' or 'causal'.
            activation: The final activation used in o = Activation(x + F(x))
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            use_layer_norm: Whether to use layer normalization in the residual layers or not.
            kwargs: Any initializers for Layer class.
        """

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
        self.branch1_layers = []
        self.branch2_layers = []
        self.layers_outputs = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)

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
            self.layers = []
            self.res_output_shape = input_shape
            with K.name_scope("temporal_branch1"):
                name = 'conv1D'
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    self.branche1_add_and_activate_layer(Conv1D(filters=self.nb_filters,
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
                self.branche1_add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))
            with K.name_scope("temporal_branch2"):
                name = 'conv1D'
                with K.name_scope(name):  # name scope used to make sure weights get unique names
                    self.branche2_add_and_activate_layer(Conv1D(filters=self.nb_filters,
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
                self.branche2_add_and_activate_layer(SpatialDropout1D(rate=self.dropout_rate))
            if not self.last_block:
                # 1x1 conv to match the shapes (channel dimension).
                name = 'conv1D'
                with K.name_scope(name):
                    # make and build this layer separately because it directly uses input_shape
                    self.shape_match_conv = Conv1D(filters=self.nb_filters,
                                                   kernel_size=1,
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
            for layer in self.layers:
                self.__setattr__(layer.name, layer)

            super(ResidualBlock, self).build(input_shape)  # done to make sure self.built is set True

    def call(self, inputs, training=None):
        """
        Returns: A tuple where the first element is the residual model tensor, and the second
                 is the skip connection tensor.
        """
        branch2_x = branch1_x = inputs
        self.layers_outputs = [branch1_x]
        for layer in self.branch1_layers:
            # inspect.signature(layer.call) got the parameters of this functiond
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            branch1_x = layer(branch1_x, training=training) if training_flag else layer(branch1_x)
            self.layers_outputs.append(branch1_x)
        for layer in self.branch2_layers:
            # inspect.signature(layer.call) got the parameters of this functiond
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

    def compute_output_shape(self, input_shape):
        return [self.res_output_shape, self.res_output_shape]


class TCN(Layer):
    """Creates a TCN layer.

        Input shape:
            A tensor of shape (batch_size, timesteps, input_dim).

        Args:
            nb_filters: The number of filters to use in the convolutional layers.
            kernel_size: The size of the kernel to use in each convolutional layer.
            dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
            nb_stacks : The number of stacks of residual blocks to use.
            padding: The padding to use in the convolutional layers, 'causal' or 'same'.
            use_skip_connections: Boolean. If we want to add skip connections from input to each residual blocK.
            return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
            activation: The activation used in the residual blocks o = Activation(x + F(x)).
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
            kernel_initializer: Initializer for the kernel weights matrix (Conv1D).
            use_batch_norm: Whether to use batch normalization in the residual layers or not.
            kwargs: Any other arguments for configuring parent class Layer. For example "name=str", Name of the model.
                    Use unique names when using multiple TCN.

        Returns:
            A TCN layer.
        """

    def __init__(self,
                 nb_filters=128,
                 kernel_size=1,
                 nb_stacks=2,
                 dilations=(1, 2, 4, 8, 16, 32, 64),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.4,
                 return_sequences=False,
                 activation='linear',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 **kwargs):

        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
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
        self.main_conv1D = None
        self.build_output_shape = None
        self.lambda_layer = None
        self.lambda_ouput_shape = None

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

        if not isinstance(nb_filters, int):
            print('An interface change occurred after the version 2.1.2.')
            print('Before: tcn.TCN(x, return_sequences=False, ...)')
            print('Now should be: tcn.TCN(return_sequences=False, ...)(x)')
            print('The alternative is to downgrade to 2.1.2 (pip install keras-tcn==2.1.2).')
            raise Exception()

        # initialize parent class
        super(TCN, self).__init__(**kwargs)

    @property
    def receptive_field(self):
        assert_msg = 'The receptive field formula works only with power of two dilations.'
        assert all([is_power_of_two(i) for i in self.dilations]), assert_msg
        return self.kernel_size * self.nb_stacks * self.dilations[-1]

    def build(self, input_shape):
        self.main_conv1D = Conv1D(filters=self.nb_filters,
                                  kernel_size=1,
                                  padding=self.padding,
                                  kernel_initializer=self.kernel_initializer)
        self.main_conv1D.build(input_shape)

        # member to hold current output shape of the layer for building purposes
        self.build_output_shape = self.main_conv1D.compute_output_shape(input_shape)

        # list to hold all the member ResidualBlocks
        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1  # cheap way to do a false case for below

        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=self.nb_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          last_block=len(self.residual_blocks) + 1 == total_num_blocks,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))
                # build newest residual block
                self.residual_blocks[-1].build(self.build_output_shape)
                self.build_output_shape = self.residual_blocks[-1].res_output_shape

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        # Author: @karolbadowski.
        output_slice_index = int(self.build_output_shape.as_list()[1] / 2) if self.padding == 'same' else -1
        self.lambda_layer = Lambda(lambda tt: tt[:, output_slice_index, :])
        self.lambda_ouput_shape = self.lambda_layer.compute_output_shape(self.build_output_shape)

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            return self.lambda_ouput_shape
        else:
            return self.build_output_shape

    def call(self, inputs, training=None):
        x = inputs
        self.layers_outputs = [x]
        try:
            x = self.main_conv1D(x)
            self.layers_outputs.append(x)
        except AttributeError:
            print('The backend of keras-tcn>2.8.3 has changed from keras to tensorflow.keras.')
            print('Either update your imports:\n- From "from keras.layers import <LayerName>" '
                  '\n- To "from tensorflow.keras.layers import <LayerName>"')
            print('Or downgrade to 2.8.3 by running "pip install keras-tcn==2.8.3"')
            import sys
            sys.exit(0)
        self.skip_connections = []
        for layer in self.residual_blocks:
            x, skip_out = layer(x, training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)

        if self.use_skip_connections:
            # skip_connect_shape = self.skip_connections
            squess_out = []
            for layer_out in self.skip_connections:
                squess_out.append(K.expand_dims(layer_out))
            concat_out = K.concatenate(squess_out)
            x = layers.Conv2D(1, 5, padding="same", activation="relu")(concat_out)
            x = K.squeeze(x, -1)
            # x = layers.add(self.skip_connections)
            self.layers_outputs.append(x)
        if not self.return_sequences:
            # 如果不需要返回的是序列，则选择其中的最后一个时间序列的输出作为最后的输出
            x = self.lambda_layer(x)
            self.layers_outputs.append(x)
        return x

    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config
