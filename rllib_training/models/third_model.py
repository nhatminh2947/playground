from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

from pommerman import constants

tf = try_import_tf()


class ThirdModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(ThirdModel, self).__init__(obs_space, action_space, num_outputs,
                                         model_config, name)

        self.board = tf.keras.layers.Input(shape=(constants.BOARD_SIZE, constants.BOARD_SIZE, 13), name="board")
        self.abilities = tf.keras.layers.Input(shape=(3,), name="abilities")

        self.norm_board = tf.keras.layers.BatchNormalization()(self.board)
        self.norm_abilities = tf.keras.layers.BatchNormalization()(self.abilities)

        self.conv2d_1 = tf.keras.layers.Conv2D(filters=32, padding="same",
                                               kernel_size=(3, 3),
                                               activation=tf.keras.activations.relu)(self.norm_board)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=64, padding="same",
                                               kernel_size=(3, 3),
                                               activation=tf.keras.activations.relu)(self.conv2d_1)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=128, padding="same",
                                               kernel_size=(3, 3),
                                               activation=tf.keras.activations.relu)(self.conv2d_2)

        self.flatten_layer = tf.keras.layers.Flatten()(self.conv2d_3)

        self.concat = tf.keras.layers.concatenate([self.flatten_layer,
                                                   self.norm_abilities])

        self.fc_1 = tf.keras.layers.Dense(units=128, name="fc_1",
                                          activation=tf.keras.activations.relu)(self.concat)
        self.fc_2 = tf.keras.layers.Dense(units=64, name="fc_2",
                                          activation=tf.keras.activations.relu)(self.fc_1)

        self.fc_2_bn = tf.keras.layers.BatchNormalization()(self.fc_2)
        self.action_layer = tf.keras.layers.Dense(units=6, name="action",
                                                  activation=tf.keras.activations.softmax)(self.fc_2_bn)
        self.value_layer = tf.keras.layers.Dense(units=1, name="value_out")(self.fc_2_bn)
        self.base_model = tf.keras.Model([self.board, self.abilities],
                                         [self.action_layer, self.value_layer])
        # self.base_model.summary()
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # print(input_dict)
        obs = input_dict["obs"]
        # print(obs)
        # print("board:", obs["board"].shape)
        # print("abilities:", obs["abilities"])
        model_out, self._value_out = self.base_model([obs["board"], obs["abilities"]])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
