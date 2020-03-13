from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

from pommerman import constants

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
        self.inputs = tf.keras.layers.Input(shape=(constants.BOARD_SIZE_ONE_VS_ONE, constants.BOARD_SIZE_ONE_VS_ONE, 3),
                                            name="inputs_11x11")
        self.position = tf.keras.layers.Input(shape=(constants.BOARD_SIZE_ONE_VS_ONE, 2), name="position")
        self.ammo = tf.keras.layers.Input(shape=(constants.NUM_ITEMS_ONE_VS_ONE,), name="ammo")
        self.can_kick = tf.keras.layers.Input(shape=(2,), name="can_kick")
        self.blast_strength = tf.keras.layers.Input(shape=(constants.NUM_ITEMS_ONE_VS_ONE,), name="blast_strength")
        self.teammate = tf.keras.layers.Input(shape=(5,), name="teammate")
        self.enemies = tf.keras.layers.Input(shape=(5, ), name="enemies")

        self.conv2d_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3))(self.inputs)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(self.conv2d_1)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(self.conv2d_2)

        self.flatten_layer = tf.keras.layers.Flatten()(self.conv2d_3)
        self.flatten_pos = tf.keras.layers.Flatten()(self.position)
        self.flatten_ammo = tf.keras.layers.Flatten()(self.ammo)
        self.flatten_kick = tf.keras.layers.Flatten()(self.can_kick)
        self.flatten_blast = tf.keras.layers.Flatten()(self.blast_strength)
        self.flatten_team = tf.keras.layers.Flatten()(self.teammate)
        self.flatten_enemies = tf.keras.layers.Flatten()(self.enemies)

        self.concat = tf.keras.layers.concatenate([self.flatten_layer,
                                                   self.flatten_pos,
                                                   self.flatten_ammo,
                                                   self.flatten_kick,
                                                   self.flatten_blast,
                                                   self.flatten_team,
                                                   self.flatten_enemies])

        self.fc_1 = tf.keras.layers.Dense(512, name="fc_1")(self.concat)
        self.fc_2 = tf.keras.layers.Dense(256, name="fc_2")(self.fc_1)

        self.action_layer = tf.keras.layers.Dense(units=6, name="action")(self.fc_2)
        self.value_layer = tf.keras.layers.Dense(units=1, name="value_out")(self.fc_2)
        self.base_model = tf.keras.Model(
            [self.inputs, self.position, self.ammo, self.can_kick, self.blast_strength, self.teammate, self.enemies],
            [self.action_layer, self.value_layer])

        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        model_out, self._value_out = self.base_model([
            tf.stack([obs["board"], obs["bomb_blast_strength"], obs["bomb_life"]], axis=-1),
            tf.stack(obs["position"], axis=-1),
            obs["ammo"],
            obs["can_kick"],
            obs["blast_strength"],
            obs["teammate"],
            obs["enemies"]])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
