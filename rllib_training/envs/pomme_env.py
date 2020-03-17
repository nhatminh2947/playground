import unittest

import gym
import ray
from gym import spaces
from ray.rllib.agents.ppo import DEFAULT_CONFIG, PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.rollout import rollout
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env

import pommerman
from pommerman import agents
from pommerman import constants
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

tf = try_import_tf()

NUM_AGENTS = 4

DICT_SPACE_FULL = spaces.Dict({
    "board": spaces.Box(low=0, high=13, shape=(11, 11)),
    "bomb_blast_strength": spaces.Box(low=0, high=13, shape=(11, 11)),
    "bomb_life": spaces.Box(low=0, high=13, shape=(11, 11)),
    "position": spaces.Tuple((spaces.Discrete(11), spaces.Discrete(11))),
    "ammo": spaces.Discrete(20),
    "can_kick": spaces.Discrete(2),
    "blast_strength": spaces.Discrete(20),
    "teammate": spaces.Discrete(5),
    "enemies": spaces.Tuple((spaces.Discrete(5), spaces.Discrete(5), spaces.Discrete(5)))
})

DICT_SPACE_1vs1 = spaces.Dict({
    "board": spaces.Box(low=0, high=13, shape=(constants.BOARD_SIZE_ONE_VS_ONE, constants.BOARD_SIZE_ONE_VS_ONE)),
    "bomb_blast_strength": spaces.Box(low=0, high=13,
                                      shape=(constants.BOARD_SIZE_ONE_VS_ONE, constants.BOARD_SIZE_ONE_VS_ONE)),
    "bomb_life": spaces.Box(low=0, high=13, shape=(constants.BOARD_SIZE_ONE_VS_ONE, constants.BOARD_SIZE_ONE_VS_ONE)),
    "position": spaces.Tuple(
        (spaces.Discrete(constants.BOARD_SIZE_ONE_VS_ONE), spaces.Discrete(constants.BOARD_SIZE_ONE_VS_ONE))),
    "ammo": spaces.Discrete(constants.NUM_ITEMS_ONE_VS_ONE),
    "can_kick": spaces.Discrete(2),
    "blast_strength": spaces.Discrete(constants.NUM_ITEMS_ONE_VS_ONE),
    "teammate": spaces.Discrete(5),
    "enemies": spaces.Discrete(5)
})

feature_keys = ["board", "bomb_blast_strength", "bomb_life", "position", "ammo", "can_kick", "blast_strength",
                "teammate", "enemies"]


class PommeFFA(gym.Env):
    def __init__(self, config=ffa_v0_fast_env()):
        global agent_id

        # config = ffa_v0_fast_env()
        env = Pomme(**config["env_kwargs"])
        num_agents = 2 if config["game_type"] == constants.GameType.OneVsOne else 4
        env.seed(0)

        agents = []
        for agent_id in range(num_agents - 1):
            agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

        agent_id += 1
        agents.append(BaseAgent(config["agent"](agent_id, config["game_type"])))

        env.set_agents(agents)
        env.set_training_agent(agents[-1].agent_id)
        env.set_init_game_state(None)

        self.action_space = spaces.Discrete(6)
        self.observation_space = DICT_SPACE_1vs1 if config["game_type"] == constants.GameType.OneVsOne \
            else DICT_SPACE_FULL
        self.env = env

    def step(self, action):
        obs = self.env.get_observations()
        actions = self.env.act(obs)
        actions.append(action)
        obs, reward, done, info = self.env.step(actions)

        return self.featurize(obs[-1]), reward[-1], done, info

    def reset(self):
        obs = self.env.reset()

        return self.featurize(obs[-1])

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def featurize(self, obs):
        feature_obs = {key: obs[key] for key in feature_keys}
        feature_obs['teammate'] = obs['teammate'].value - constants.Item.AgentDummy.value
        feature_obs['enemies'] = obs['enemies'][0].value - constants.Item.AgentDummy.value
        return feature_obs


class PommeMultiAgent(MultiAgentEnv):
    def __init__(self, config):
        self.agent_list = [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ]
        self._step_count = 0
        self.agent_names = config["agent_names"]
        self.env = pommerman.make(config["env_id"], self.agent_list)
        self.phase = config["phase"]
        print("Phase", self.phase)
        self.dones = set()
        self._max_steps = self.env._max_steps
        self._base_reward = 0.001

        self.action_space = self.env.action_space
        self.observation_space = DICT_SPACE_FULL

    def step(self, action_dict):
        # self.env.render()
        obs = self.env.get_observations()
        actions = self.env.act(obs)

        for agent_name in self.agent_names:
            if agent_name not in action_dict.keys():
                action_dict[agent_name] = 0

        actions = [actions[0], action_dict[self.agent_names[0]],
                   actions[2], action_dict[self.agent_names[1]]]

        _obs, _reward, _done, _info = self.env.step(actions)

        dones = {"__all__": _done}
        obs = {}
        rewards = {}
        infos = {}

        for agent_id, agent_name in zip([1, 3], self.agent_names):
            if self.env._agents[agent_id].is_alive:
                dones[agent_name] = False
                obs[agent_name] = self.featurize(_obs[agent_id])
                rewards[agent_name] = self._base_reward + self._get_rewards(_done, _info["result"], agent_id)
                infos[agent_name] = {info_k: info_v for info_k, info_v in _info.items()}
            elif agent_id not in self.dones:
                self.dones.add(agent_id)
                dones[agent_name] = True
                obs[agent_name] = self.featurize(_obs[agent_id])
                rewards[agent_name] = -1
                infos[agent_name] = {info_k: info_v for info_k, info_v in _info.items()}

        self._step_count += 1
        return obs, rewards, dones, infos

    def _get_rewards(self, done, result, agent_id):
        if self.phase == 0:
            if self._step_count >= self._max_steps:
                return 1 if self.env._agents[agent_id].is_alive else -1

            return 0
        else:
            if done and result == constants.Result.Tie:
                return -1
            elif done and result == constants.Result.Win:
                return 1
            return 0

    def featurize(self, obs):
        feature_obs = {key: obs[key] for key in feature_keys}
        feature_obs['teammate'] = obs['teammate'].value - constants.Item.AgentDummy.value
        feature_obs['enemies'] = [enemy.value - constants.Item.AgentDummy.value for enemy in obs['enemies']]

        return feature_obs

    def set_phase(self, phase):
        if phase == 1:
            print("Phase 1")
            self.agent_list = [
                agents.StaticAgent(),
                agents.StaticAgent(),
                agents.StaticAgent(),
                agents.StaticAgent(),
            ]
            self.phase = 1
            self.env = pommerman.make("Blank-PommeTeam-v0", self.agent_list)
            self.env.reset()
        elif phase == 2:
            print("Phase 2")
            self.agent_list = [
                agents.StaticAgent(),
                agents.StaticAgent(),
                agents.StaticAgent(),
                agents.StaticAgent(),
            ]
            self.phase = 1
            self.env = pommerman.make("PommeTeam-v0", self.agent_list)
            self.env.reset()
        elif phase == 3:
            print("Phase 3")
            self.agent_list = [
                agents.SmartRandomAgentNoBomb(),
                agents.StaticAgent(),
                agents.SmartRandomAgentNoBomb(),
                agents.StaticAgent(),
            ]
            self.phase = 2
            self.env = pommerman.make("PommeTeam-v0", self.agent_list)
            self.env.reset()

    def reset(self):
        self.dones.clear()
        obs = self.env.reset()
        self._step_count = 0
        obs = {self.agent_names[0]: self.featurize(obs[1]),
               self.agent_names[1]: self.featurize(obs[3])}
        return obs


class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        print(model_config)
        print(obs_space.original_space)
        # self.model = FullyConnectedNetwork(obs_space, action_space,
        #                                    num_outputs, model_config, name)
        # self.register_variables(self.model.variables())

        self.inputs = tf.keras.layers.Input(shape=(11, 11, 3), name="observations")
        self.conv2d_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(11, 11, 3))(self.inputs)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(self.conv2d_1)
        self.conv2d_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3))(self.conv2d_2)
        self.flatten_layer = tf.keras.layers.Flatten()(self.conv2d_3)
        self.final_layer = tf.keras.layers.Dense(256, name="final_layer")(self.flatten_layer)
        self.action_layer = tf.keras.layers.Dense(units=6, name="action")(self.final_layer)
        self.value_layer = tf.keras.layers.Dense(units=1, name="value_out")(self.final_layer)
        self.base_model = tf.keras.Model(self.inputs, [self.action_layer, self.value_layer])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        print("input_dict", input_dict)
        print("seq_lens", seq_lens)
        obs = tf.stack(
            [input_dict["obs"]["board"], input_dict["obs"]["bomb_blast_strength"], input_dict["obs"]["bomb_life"]],
            axis=-1)

        print(obs)
        model_out, self._value_out = self.base_model(tf.stack(
            [input_dict["obs"]["board"], input_dict["obs"]["bomb_blast_strength"], input_dict["obs"]["bomb_life"]],
            axis=-1))
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class PommeFFATest(unittest.TestCase):
    def testEnvironment(self):
        ModelCatalog.register_custom_model("my_model", CustomModel)

        config = DEFAULT_CONFIG.copy()
        config['num_workers'] = 0
        # config['model']['fcnet_hiddens'] = [100, 100]
        config['env_config'] = ffa_v0_fast_env()

        register_env("PommeFFA", lambda _: PommeFFA())
        agent = PPOTrainer(env="PommeFFA", config=config)
        agent.train()
        path = agent.save()
        agent.stop()

        # Test train works on restore
        agent2 = PPOTrainer(env="PommeFFA", config=config)
        agent2.restore(path)
        agent2.train()

        # Test rollout works on restore
        rollout(agent2, "PommeFFA", 100)


# class NestedSpacesTest(unittest.TestCase):
#     def testStep(self):
#         ModelCatalog.register_custom_model("my_model", CustomModel)
#
#         env_config = ffa_competition_env()
#         env = Pomme(**env_config['env_kwargs'])
#         obs_space = DICT_SPACE
#         act_space = env.action_space
#
#         env_config["model"] = {
#             "model": {
#                 "custom_model": "simple_model"
#             }
#         }
#
#         config = DEFAULT_CONFIG.copy()
#         config['num_workers'] = 0
#         # config['model']['fcnet_hiddens'] = [100, 100]
#         config['env_config'] = ffa_v0_fast_env()
#         config['multiagent'] = {
#             # Map from policy ids to tuples of (policy_cls, obs_space,
#             # act_space, config). See rollout_worker.py for more info.
#             "policies": {
#                 "simple": (SimplePolicy, obs_space, act_space, {"obs_space": obs_space}),
#                 "ppo_policy": (PPOTFPolicy, obs_space, act_space, {
#                     "model": {
#                         "custom_model": "my_model"
#                     }
#                 }),
#             },
#             # Function mapping agent ids to policy ids.
#             "policy_mapping_fn": (lambda agent_id: "ppo_policy" if agent_id.startswith('ppo') else "simple"),
#             # Optional whitelist of policies to train, or None for all policies.
#             "policies_to_train": ["ppo_policy"],
#         }
#
#         register_env("PommeRllib", lambda _: PommeRllib(env_config))
#         agent = PPOTrainer(env="PommeRllib", config=config)
#         agent.train()
#         path = agent.save()
#         agent.stop()
#
#         # Test train works on restore
#         agent2 = PPOTrainer(env="PommeRllib", config=config)
#         agent2.restore(path)
#         agent2.train()
#
#         # Test rollout works on restore
#         rollout(agent2, "PommeRllib", 100)


if __name__ == "__main__":
    ray.init(num_cpus=5)
    unittest.main(verbosity=2)
