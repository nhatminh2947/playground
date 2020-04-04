import numpy as np
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

from pommerman import constants

tf = try_import_tf()


class BaseModel(tf.keras.Model):
    def __init__(self, initializer=tf.keras.initializers.glorot_uniform, init_seed=None):
        super(BaseModel, self).__init__(name="base")

        self.conv2d_32 = tf.keras.layers.Conv2D(filters=32, padding="same",
                                                kernel_size=(3, 3),
                                                kernel_initializer=initializer(seed=init_seed),
                                                use_bias=False,
                                                activation=tf.keras.activations.relu)
        self.conv2d_64 = tf.keras.layers.Conv2D(filters=64, padding="same",
                                                kernel_size=(3, 3),
                                                kernel_initializer=initializer(seed=init_seed),
                                                use_bias=False,
                                                activation=tf.keras.activations.relu)
        self.conv2d_128 = tf.keras.layers.Conv2D(filters=128, padding="same",
                                                 kernel_size=(3, 3),
                                                 kernel_initializer=initializer(seed=init_seed),
                                                 use_bias=False,
                                                 activation=tf.keras.activations.relu)

        self.flatten_layer = tf.keras.layers.Flatten()

        self.fc_128 = tf.keras.layers.Dense(units=128, name="fc_1",
                                            kernel_initializer=initializer(seed=init_seed),
                                            use_bias=False,
                                            activation=tf.keras.activations.relu)

        self.fc_64 = tf.keras.layers.Dense(units=64, name="fc_2",
                                           kernel_initializer=initializer(seed=init_seed),
                                           use_bias=False,
                                           activation=tf.keras.activations.relu)

        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv2d_32(inputs)
        x = self.conv2d_64(x)
        x = self.conv2d_128(x)
        x = self.flatten_layer(x)
        x = self.fc_128(x)
        x = self.fc_64(x)
        x = self.bn(x)

        return x


class RunningStats(object):
    # This class which computes global stats is adapted & modified from:
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        # self.std = np.sqrt(np.maximum(self.var, 1e-2))
        self.count = batch_count + self.count


class RNDModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(RNDModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.base_model = BaseModel()

        self.board = tf.keras.layers.Input(shape=(constants.BOARD_SIZE, constants.BOARD_SIZE, 16), name="board")
        self.output = self.base_model.call(self.board)

        #
        # self.norm_board = tf.keras.layers.BatchNormalization()(self.board)
        #
        # self.conv2d_1 = tf.keras.layers.Conv2D(filters=32, padding="same",
        #                                        kernel_size=(3, 3),
        #                                        activation=tf.keras.activations.relu)(self.norm_board)
        # self.conv2d_2 = tf.keras.layers.Conv2D(filters=64, padding="same",
        #                                        kernel_size=(3, 3),
        #                                        activation=tf.keras.activations.relu)(self.conv2d_1)
        # self.conv2d_3 = tf.keras.layers.Conv2D(filters=128, padding="same",
        #                                        kernel_size=(3, 3),
        #                                        activation=tf.keras.activations.relu)(self.conv2d_2)
        #
        # self.flatten_layer = tf.keras.layers.Flatten()(self.conv2d_3)
        #
        # self.fc_1 = tf.keras.layers.Dense(units=128, name="fc_1",
        #                                   activation=tf.keras.activations.relu)(self.flatten_layer)
        # self.fc_2 = tf.keras.layers.Dense(units=64, name="fc_2",
        #                                   activation=tf.keras.activations.relu)(self.fc_1)
        #
        # self.fc_2_bn = tf.keras.layers.BatchNormalization()(self.fc_2)

        self.action_layer = tf.keras.layers.Dense(units=6, name="action",
                                                  activation=tf.keras.activations.softmax)(self.output)
        self.value_layer = tf.keras.layers.Dense(units=1, name="value_out")(self.output)

        self.actor_model = tf.keras.Model(self.board, [self.action_layer, self.value_layer])

        self.register_variables(self.actor_model.variables)

        # self.value_layer = tf.keras.layers.Dense(units=1, name="value_out")(self.fc_2_bn)
        #
        # self.critic_model = tf.keras.Model(self.board, self.value_layer)
        #
        # self.register_variables(self.critic_model.variables)

        self.target_model = BaseModel(init_seed=2947)

    def forward(self, input_dict, state, seq_lens):
        # print(input_dict)
        obs = input_dict["obs"]
        # print("abilities:", obs["abilities"])
        model_out, self._value_out = self.actor_model(obs["board"])

        # self.curiosity_loss = self.get_curiosity(obs["board"])
        # self._value_out = self.critic_model(obs["board"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
