import ray
from ray import tune
from ray import rllib
from ray.tune import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.policy import Policy, TFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.models import FullyConnectedNetwork, Model, ModelCatalog

from pommerman import configs
from pommerman import constants
from pommerman import make
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, BaseAgent

from training.policies import SimplePolicy
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete, Box

import numpy as np
import gym

agent_names = ["simple_agent_0", "simple_agent_1", "simple_agent_2", "ppo_agent_3"]
NUM_AGENTS = 4


def create_obs_space():
    return Dict({
        "board": Box(low=0, high=13, shape=(11, 11)),
        "bomb_blast_strength": Box(low=0, high=13, shape=(11, 11)),
        "bomb_life": Box(low=0, high=13, shape=(11, 11)),
        "position": Tuple((Discrete(11), Discrete(11))),
        "ammo": Discrete(constants.NUM_ITEMS),
        "can_kick": Discrete(2),
        "blast_strength": Discrete(constants.NUM_ITEMS),
        "teammate": Discrete(14),
        "enemies": MultiDiscrete([14, 14, 14])
    })


keys = ["board", "bomb_blast_strength", "bomb_life", "position", "ammo", "can_kick", "blast_strength", "teammate",
        "enemies"]


class PommeRllib(MultiAgentEnv):

    def __init__(self, config):
        self.env = Pomme(**config["env_kwargs"])
        self.env.seed(2947)
        agents = []
        for agent_id in range(3):
            agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))
        agents.append(BaseAgent(config["agent"](3, config["game_type"])))

        self.env.set_agents(agents)
        self.env.set_init_game_state(None)
        self.dones = set()

        self.observation_space = self.env.observation_space

    def step(self, action_dict):
        # self.env.render()
        actions = {agent_name: 0 for agent_name in agent_names}
        actions.update(action_dict)
        # print('action_dict:', action_dict)
        # print('actions:', actions)
        # print('current_obs:', self.env.observations)
        _obs, _reward, _done, _info = self.env.step(list(actions.values()))

        dones = {"__all__": _done}
        obs = {}
        rewards = {}
        infos = {}

        for id, agent in enumerate(self.env._agents):
            if agent.is_alive:
                dones[agent_names[id]] = False
                obs[agent_names[id]] = self.env.featurize(_obs[id])
                rewards[agent_names[id]] = _reward[id]
                infos[agent_names[id]] = {info_k: info_v for info_k, info_v in _info.items()}
            elif not agent.is_alive and id not in self.dones:
                self.dones.add(id)
                dones[agent_names[id]] = True
                obs[agent_names[id]] = self.env.featurize(_obs[id])
                rewards[agent_names[id]] = _reward[id]
                infos[agent_names[id]] = {info_k: info_v for info_k, info_v in _info.items()}

            if id == 3 and not agent.is_alive:
                for id, agent in enumerate(self.env._agents):
                    if id not in self.dones:
                        dones[agent_names[id]] = True
                dones["__all__"] = True

        # print(dones)

        return obs, rewards, dones, infos

    def reset(self):
        self.dones.clear()
        obs = self.env.reset()
        obs = {agent_names[i]: self.env.featurize(obs[i]) for i in range(NUM_AGENTS)}
        return obs


def test_env():
    env = PommeRllib(configs.ffa_v0_fast_env())
    env.reset()

    obs, reward, done, info = env.step({agent_names[i]: 0 for i in range(NUM_AGENTS)})

    # obs = env.featurize(obs)
    # print('obs[0].size={}'.format(obs['simple_agent_0'].size))
    # print(obs['simple_agent_0'])
    print('obs={}'.format(obs))
    print('reward={}'.format(reward))
    print('done={}'.format(done))
    print('info={}'.format(info))


if __name__ == "__main__":
    ray.shutdown()
    ray.init(local_mode=False)

    config = configs.ffa_v0_fast_env()

    env = Pomme(**config['env_kwargs'])

    obs_space = env.observation_space
    act_space = env.action_space

    tune.run(
        "PPO",
        stop={
            "training_iteration": 10000,
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        config={
            "env": PommeRllib,
            "num_workers": 9,
            "num_gpus": 1,
            "env_config": config,
            # === Settings for Multi-Agent Environments ===
            "multiagent": {
                # Map from policy ids to tuples of (policy_cls, obs_space,
                # act_space, config). See rollout_worker.py for more info.
                "policies": {
                    "simple": (SimplePolicy, obs_space, act_space, config['env_kwargs']),
                    "ppo_policy": (PPOTFPolicy, obs_space, act_space, {}),
                },
                # Function mapping agent ids to policy ids.
                "policy_mapping_fn": (lambda agent_id: "ppo_policy" if agent_id.startswith('ppo') else "simple"),
                # Optional whitelist of policies to train, or None for all policies.
                "policies_to_train": ["ppo_policy"],
            },
            "log_level": "WARN",
        }
    )
