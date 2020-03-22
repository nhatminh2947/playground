import unittest

import gym
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf

import pommerman
from pommerman import agents
from pommerman import constants
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

tf = try_import_tf()

NUM_AGENTS = 4

DICT_SPACE_FULL = spaces.Dict({
    "board": spaces.Box(low=0, high=20, shape=(11, 11, 13)),
    "abilities": spaces.Box(low=0, high=20, shape=(3,))
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

        self.dones = set()
        self._max_steps = self.env._max_steps
        self._base_reward = 0.001

        self.action_space = self.env.action_space
        self.observation_space = DICT_SPACE_FULL

    def render(self):
        self.env.render()

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
                rewards[agent_name] = self._base_reward + self._get_rewards(_done, _info["result"])
                infos[agent_name] = {info_k: info_v for info_k, info_v in _info.items()}
            elif agent_id not in self.dones:
                self.dones.add(agent_id)
                dones[agent_name] = True
                obs[agent_name] = self.featurize(_obs[agent_id])
                rewards[agent_name] = -1
                infos[agent_name] = {info_k: info_v for info_k, info_v in _info.items()}

        self._step_count += 1
        return obs, rewards, dones, infos

    def _get_rewards(self, done, result):
        if done and result == constants.Result.Tie:
            return 0
        elif done and result == constants.Result.Win:
            return 1
        return 0

    def featurize(self, obs):
        # print(obs)
        id = 0
        features = {"board": np.zeros(shape=(11, 11, 13))}
        # print(features)
        for item in constants.Item:
            if item in [constants.Item.Bomb,
                        constants.Item.Flames,
                        constants.Item.Agent0,
                        constants.Item.Agent1,
                        constants.Item.Agent2,
                        constants.Item.Agent3,
                        constants.Item.AgentDummy]:
                continue
            # print("item:", item)
            # print("board:", obs["board"])

            features["board"][:, :, id][obs["board"] == item.value] = 1
            id += 1
        # print(id)
        features["board"][:, :, id] = obs["flame_life"]
        id += 1

        features["board"][:, :, id] = obs["bomb_life"]
        id += 1

        features["board"][:, :, id] = obs["bomb_blast_strength"]
        id += 1

        features["board"][:, :, id][obs["position"]] = 1
        id += 1

        features["board"][:, :, id][obs["board"] == obs["teammate"].value] = 1
        id += 1

        for enemy in obs["enemies"]:
            features["board"][:, :, id][obs["board"] == enemy.value] = 1
        id += 1

        # print("id:", id)
        features["abilities"] = np.asarray([obs["ammo"], obs["blast_strength"], obs["can_kick"]], dtype=np.float)

        return features

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


class TestPommeEnv(unittest.TestCase):
    def test_featurize_function(self):
        agents_list = [agents.StaticAgent(),
                       agents.StaticAgent(),
                       agents.StaticAgent(),
                       agents.StaticAgent()]

        env = pommerman.make("Mines-PommeTeam-v0", agents_list)

        obs = env.reset()

        pomme_env = PommeMultiAgent({
            "agent_names": [],
            "env_id": "Mines-PommeTeam-v0",
            "phase": 0
        })

        print(pomme_env.featurize(obs[0]))
