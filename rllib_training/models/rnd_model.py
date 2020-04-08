import numpy as np
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

from pommerman import constants

tf = try_import_tf()


class Representation(tf.keras.Model):
    def __init__(self, initializer=tf.keras.initializers.glorot_uniform, init_seed=None):
        super(Representation, self).__init__(name='base')

        self.conv2d_32 = tf.keras.layers.Conv2D(filters=32, padding='same',
                                                kernel_size=(3, 3),
                                                kernel_initializer=initializer(seed=init_seed),
                                                use_bias=False,
                                                activation=tf.keras.activations.relu)
        self.conv2d_64 = tf.keras.layers.Conv2D(filters=64, padding='same',
                                                kernel_size=(3, 3),
                                                kernel_initializer=initializer(seed=init_seed),
                                                use_bias=False,
                                                activation=tf.keras.activations.relu)
        self.conv2d_128 = tf.keras.layers.Conv2D(filters=128, padding='same',
                                                 kernel_size=(3, 3),
                                                 kernel_initializer=initializer(seed=init_seed),
                                                 use_bias=False,
                                                 activation=tf.keras.activations.relu)

        self.flatten_layer = tf.keras.layers.Flatten()

        self.fc_128 = tf.keras.layers.Dense(units=128, name='fc_1',
                                            kernel_initializer=initializer(seed=init_seed),
                                            use_bias=False,
                                            activation=tf.keras.activations.relu)

        self.fc_64 = tf.keras.layers.Dense(units=64, name='fc_2',
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


class CNNModel(TFModelV2):
    '''Example of a custom model that just delegates to a fc-net.'''

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CNNModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.board = tf.keras.layers.Input(shape=(constants.BOARD_SIZE, constants.BOARD_SIZE, 16), name='board')

        self.features = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, name='fc_1',
                                  use_bias=False,
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(units=64, name='fc_1',
                                  use_bias=False,
                                  activation=tf.keras.activations.relu)
        ])(self.board)

        self.action_layer = tf.keras.layers.Dense(units=6, name='action',
                                                  activation=tf.keras.activations.softmax)(self.features)

        self.critic_intrinsic = tf.keras.layers.Dense(units=1, name='intrinsic_value')(self.features)

        self.critic_extrinsic = tf.keras.layers.Dense(units=1, name='extrinsic_value')(self.features)

        self.cnn_model = tf.keras.Model(self.board, [self.action_layer, self.critic_intrinsic, self.critic_extrinsic])

        # RND Model
        self.target = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu, trainable=False),
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu, trainable=False),
            tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu, trainable=False),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, name='fc_1',
                                  use_bias=False,
                                  activation=tf.keras.activations.relu, trainable=False),
            tf.keras.layers.Dense(units=64, name='fc_1',
                                  use_bias=False,
                                  activation=tf.keras.activations.relu, trainable=False)
        ])

        self.predictor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=(3, 3), use_bias=False,
                                   activation=tf.keras.activations.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, name='fc_1',
                                  use_bias=False,
                                  activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(units=64, name='fc_1',
                                  use_bias=False,
                                  activation=tf.keras.activations.relu)
        ])

        self.target(inputs=self.board)
        self.predictor(inputs=self.board)

        self.register_variables(self.cnn_model.variables)
        self.register_variables(self.target.variables)
        self.register_variables(self.predictor.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        model_out, self._intrinsic_value, self._extrinsic_value = self.cnn_model(obs['board'])
        print('self._intrinsic_value', tf.reshape(self._extrinsic_value, [-1]))

        return model_out, state

    def value_function(self):
        return tf.reshape(self._extrinsic_value + self._intrinsic_value, [-1])

    def extrinsic_value_function(self):
        return tf.reshape(self._extrinsic_value, [-1])

    def intrinsic_value_function(self):
        return tf.reshape(self._intrinsic_value, [-1])

    def compute_intrinsic_reward(self, next_obs):
        print('next_obs', type(next_obs))
        print(next_obs)
        next_obs = tf.reshape(next_obs, [-1, 11, 11, 16])
        print(next_obs)
        target_next_feature = self.target(next_obs)
        predict_next_feature = self.predictor(next_obs)
        intrinsic_reward = tf.keras.losses.mean_squared_error(target_next_feature, predict_next_feature)

        print('intrinsic_reward', intrinsic_reward)

        return tf.reshape(intrinsic_reward, [-1])
