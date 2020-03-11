from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class FirstModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(FirstModel, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)
        # self.model = FullyConnectedNetwork(obs_space, action_space,
        #                                    num_outputs, model_config, name)
        # self.register_variables(self.model.variables())

        self.inputs = tf.keras.layers.Input(shape=(11, 11, 3), name="inputs_11x11")
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(11, 11, 3))(self.inputs)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(self.conv2d_1)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(self.conv2d_2)

        # self.info = tf.keras.layers.Input(shape=(9,), name="info")
        self.flatten_layer = tf.keras.layers.Flatten()(self.conv2d_3)

        self.final_layer = tf.keras.layers.Dense(256, name="final_layer")(self.flatten_layer)
        self.action_layer = tf.keras.layers.Dense(units=6, name="action")(self.final_layer)
        self.value_layer = tf.keras.layers.Dense(units=1, name="value_out")(self.final_layer)
        self.base_model = tf.keras.Model([self.inputs], [self.action_layer, self.value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(tf.stack(
            [input_dict["obs"]["board"], input_dict["obs"]["bomb_blast_strength"], input_dict["obs"]["bomb_life"]],
            axis=-1))
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
