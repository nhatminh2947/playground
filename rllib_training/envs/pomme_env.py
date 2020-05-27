import unittest

import gym
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pommerman import utility
import pommerman
from pommerman import agents
from pommerman import constants


class Ability:
    def __init__(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False

    def reset(self):
        self.ammo = 1
        self.blast_strength = 2
        self.can_kick = False


class PommeMultiAgent(MultiAgentEnv):
    def __init__(self, config):
        self.agent_list = [
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
            agents.StaticAgent(),
        ]
        self.is_render = config["render"]
        self._step_count = 0
        self.env = pommerman.make(config["env_id"], self.agent_list)
        self._max_steps = self.env._max_steps
        self.action_space = self.env.action_space
        self.eliminated = []
        self.alive_agents = [0, 1, 2, 3]
        self.ability = [Ability()] * 4
        self.fow = {i: np.zeros(shape=(16, 11, 11)) for i in range(4)}

    def render(self):
        self.env.render()

    def step(self, action_dict):
        if self.is_render:
            self.render()
        prev_obs = self.env.get_observations()

        actions = []
        for id in range(4):
            if id in action_dict:
                actions.append(action_dict[id])
            else:
                actions.append(0)

        _obs, _reward, _done, _info = self.env.step(actions)

        obs = {}
        rewards = {}
        dones = {}
        infos = {}

        # print('_done', _done)

        for id in self.alive_agents:
            obs[id] = self.featurize(_obs[id])
            rewards[id] = self.reward_shaping(id, _obs[id], prev_obs[id]['board'], _info)
            infos[id] = _info

            if (id + 10) not in _obs[id]['alive']:
                dones[id] = True
            else:
                dones[id] = False

        self.update_fow(obs)

        dones["__all__"] = _done

        self._step_count += 1

        self.alive_agents = np.asarray(_obs[0]['alive']) - constants.Item.Agent0.value

        self.eliminated = []
        for i in range(4):
            if i not in self.alive_agents:
                self.eliminated.append(i)

        return self.fow, rewards, dones, infos

    def update_fow(self, obs):
        for i in range(4):
            if i in list(obs.keys()):
                for channel in range(16):
                    if channel < 9:
                        self.fow[i][channel][obs[i][channel] != 0] = obs[i][channel][obs[i][channel] != 0]
                        print(self.fow[i][channel])
                    else:
                        self.fow[i][channel] = obs[i][channel]
                        print(self.fow[i][channel])
            else:
                self.fow.pop(i, None)

    def reward_shaping(self, agent_id, new_obs, prev_board, info):
        reward = 0
        current_alive_agents = np.asarray(new_obs['alive']) - constants.Item.Agent0.value

        if info['result'] == constants.Result.Tie:
            return -1

        if agent_id not in current_alive_agents:
            return -1

        if agent_id % 2 == 0:
            enemies = [1, 3]
        else:
            enemies = [0, 2]

        if utility.position_is_powerup(prev_board, new_obs['position']):
            if constants.Item(prev_board[new_obs['position']]) == constants.Item.IncrRange:
                reward += 0.01
                self.ability[agent_id].blast_strength += 1
            elif constants.Item(prev_board[new_obs['position']]) == constants.Item.ExtraBomb:
                reward += 0.01
                self.ability[agent_id].ammo += 1
            elif not self.ability[agent_id].can_kick and constants.Item(
                    prev_board[new_obs['position']]) == constants.Item.Kick:
                reward += 0.05
                self.ability[agent_id].can_kick = True

        for enemy in enemies:
            if enemy not in current_alive_agents and enemy not in self.eliminated:
                reward += 0.5

        return reward

    # Meaning of channels
    # 0 passage             fow
    # 1 Rigid               fow
    # 2 Wood                fow
    # 3 ExtraBomb           fow
    # 4 IncrRange           fow
    # 5 Kick                fow
    # 6 FlameLife           fow
    # 7 BombLife            fow
    # 8 BombBlastStrength   fow
    # 9 Fog
    # 10 Position
    # 11 Teammate
    # 12 Enemies
    # 13 Ammo
    # 14 BlastStrength
    # 15 CanKick
    def featurize(self, obs):
        id = 0
        features = np.zeros(shape=(16, 11, 11))
        # print(obs['board'])
        for item in constants.Item:
            if item in [constants.Item.Bomb,
                        constants.Item.Fog,
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

        features[id, :, :][obs["board"] == constants.Item.Fog.value] = 1
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

        # for i in range(id):
        #     print(features[i, :, :])

        return features

    def reset(self):
        _obs = self.env.reset()
        self._step_count = 0
        self.eliminated = []
        self.alive_agents = [0, 1, 2, 3]
        obs = {}
        self.fow = {i: np.zeros(shape=(16, 11, 11)) for i in range(4)}

        for i in range(4):
            self.ability[i].reset()
            obs[i] = self.featurize(_obs[i])

        return obs
