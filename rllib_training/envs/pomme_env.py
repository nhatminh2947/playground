import unittest

import gym
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import pommerman
from pommerman import agents
from pommerman import constants
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

NUM_AGENTS = 4

DICT_SPACE_FULL = spaces.Dict({
    "board": spaces.Box(low=0, high=20, shape=(11, 11, 13)),
    "abilities": spaces.Box(low=0, high=20, shape=(3,))
})

feature_keys = ["board", "bomb_blast_strength", "bomb_life", "position", "ammo", "can_kick", "blast_strength",
                "teammate", "enemies"]


class PommeMultiAgent(MultiAgentEnv):
    def __init__(self, config):
        self.agent_list = [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ]
        self._step_count = 0
        self.env = pommerman.make(config["env_id"], self.agent_list)

        self._max_steps = self.env._max_steps

        self.action_space = self.env.action_space
        self.observation_space = DICT_SPACE_FULL
        self._agent_ids = [[0, 2], [1, 3]]

    def render(self):
        self.env.render()

    def step(self, action_dict):
        obs = self.env.get_observations()
        actions = self.env.act(obs)
        print('action_dict', action_dict)
        for agent_name in self.agent_names:
            if agent_name not in action_dict.keys():
                action_dict[agent_name] = 0

        if self._position:
            actions = [actions[0], action_dict[self.agent_names[0]],
                       actions[2], action_dict[self.agent_names[1]]]
        else:
            actions = [action_dict[self.agent_names[0]], actions[1],
                       action_dict[self.agent_names[1]], actions[3]]

        _obs, _reward, _done, _info = self.env.step(actions)

        dones = {"__all__": _done}
        obs = {0: self.featurize(_obs[0]),
               1: self.featurize(_obs[1]),
               2: self.featurize(_obs[2]),
               3: self.featurize(_obs[3])}
        rewards = {0: 0,
                   1: 0,
                   2: 0,
                   3: 0}
        infos = {}

        self._step_count += 1
        return obs, rewards, dones, infos

    def _get_infos(self, done, info):
        if done:
            if info["result"] == constants.Result.Win:
                if info["winners"] == self._agent_ids[self._position]:
                    return {"result": constants.Result.Win}
                else:
                    return {"result": constants.Result.Loss}
            else:
                return {"result": constants.Result.Tie}
        return {"result": constants.Result.Incomplete}

    def _get_rewards(self, done, result):
        if done and result == constants.Result.Tie:
            return 0
        elif done and result == constants.Result.Win:
            return 1
        return 0

    def featurize(self, obs):
        id = 0
        features = np.zeros(shape=(16, 11, 11))

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

            features[id, :, :][obs["board"] == item.value] = 1
            id += 1

        for feature in ["flame_life", "bomb_life", "bomb_blast_strength"]:
            features[id, :, :] = obs[feature]
            id += 1

        features[id, :, :][obs["position"]] = 1
        id += 1

        features[id, :, :][obs["board"] == obs["teammate"].value] = 1
        id += 1

        for enemy in obs["enemies"]:
            features[id, :, :][obs["board"] == enemy.value] = 1
        id += 1

        features[id, :, :] = np.full(shape=(11, 11), fill_value=obs["ammo"])
        id += 1

        features[id, :, :] = np.full(shape=(11, 11), fill_value=obs["blast_strength"])
        id += 1

        features[id, :, :] = np.full(shape=(11, 11), fill_value=(1 if obs["can_kick"] else 0))
        id += 1

        return features


    def reset(self):
        obs = self.env.reset()
        self._step_count = 0

        obs = {0: self.featurize(obs[0]),
               1: self.featurize(obs[1]),
               2: self.featurize(obs[2]),
               3: self.featurize(obs[3])}

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
