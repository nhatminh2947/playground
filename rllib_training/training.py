import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from gym import spaces
import pommerman
from pommerman import constants
from rllib_training import models
from rllib_training.envs import pomme_env
from rllib_training.envs.pomme_env import PommeMultiAgent
from rllib_training.policies import RandomPolicy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from typing import Dict


class PommeCallbacks(DefaultCallbacks):
    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        last_info = None

        for agent_name in range(4):
            if episode.last_info_for(agent_name)["result"] != constants.Result.Incomplete:
                last_info = episode.last_info_for(agent_name)
                break
        # print(episode.agent_rewards)
        # print(episode._agent_reward_history["ppo_agent_1"])
        # print(episode._agent_reward_history["ppo_agent_2"])
        if "win_team_1" not in episode.custom_metrics:
            episode.custom_metrics["win_team_1"] = 0
        if "win_team_2" not in episode.custom_metrics:
            episode.custom_metrics["win_team_2"] = 0
        if "tie" not in episode.custom_metrics:
            episode.custom_metrics["tie"] = 0
        # print(last_info)
        if last_info["result"] == constants.Result.Win:
            if last_info['winners'] == [0, 2]:
                episode.custom_metrics["win_team_1"] += 1
            else:
                episode.custom_metrics["win_team_2"] += 1
        elif last_info["result"] == constants.Result.Tie:
            episode.custom_metrics["tie"] += 1

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        for agent_name in range(4):
            action = episode.last_action_for(agent_name)
            if action == constants.Action.Bomb.value:
                episode.custom_metrics["bomb_agent_{}".format(agent_name)] += 1

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        for agent_name in range(4):
            episode.custom_metrics["bomb_agent_{}".format(agent_name)] = 0
            episode.custom_metrics["reward_agent_{}".format(agent_name)] = 0


def training_team(params):
    # env_id = "PommeTeamCompetition-v0"
    # env_id = "PommeFFACompetitionFast-v0"
    env_id = "PommeTeam-v0"

    env_config = {
        "env_id": env_id,
        "render": params['render']
    }

    env = pommerman.make(env_id, [])
    obs_space = spaces.Box(low=0, high=20, shape=(16, 11, 11))
    act_space = env.action_space

    ModelCatalog.register_custom_model("1st_model", models.FirstModel)
    ModelCatalog.register_custom_model("2nd_model", models.SecondModel)
    ModelCatalog.register_custom_model("torch_conv", models.ActorCriticModel)

    tune.register_env("PommeMultiAgent-v0", lambda x: PommeMultiAgent(env_config))

    def gen_policy():
        config = {
            "model": {
                "custom_model": "torch_conv",
                "custom_options": {
                    "in_channels": 16,
                    "feature_dim": 512
                }
            },
            "gamma": params['gamma'],
            "use_pytorch": True
        }
        return PPOTorchPolicy, obs_space, act_space, config

    policies = {
        "policy_{}".format(i): gen_policy() for i in range(2)
    }
    policies["random"] = (RandomPolicy, obs_space, act_space, {})
    print(policies.keys())

    trials = tune.run(
        PPOTrainer,
        name=params["name"],
        queue_trials=True,
        stop={
            "training_iteration": params["training_iteration"],
            # "timesteps_total": 100000000
        },
        checkpoint_freq=params["checkpoint_freq"],
        checkpoint_at_end=True,
        config={
            "lr": params["lr"],
            "entropy_coeff": params["entropy_coeff"],
            "kl_coeff": 0.0,  # disable KL
            "batch_mode": "truncate_episodes" if params['truncate_episodes'] else 'complete_episodes',
            "rollout_fragment_length": params['rollout_fragment_length'],
            "env": PommeMultiAgent,
            "env_config": env_config,
            "num_workers": params['num_workers'],
            "num_envs_per_worker": params['num_envs_per_worker'],
            "num_gpus_per_worker": params['num_gpus_per_worker'],
            "num_gpus": params['num_gpus'],
            "train_batch_size": params['train_batch_size'],
            "sgd_minibatch_size": params['sgd_minibatch_size'],
            "clip_param": params['clip_param'],
            "lambda": params['lambda'],
            "num_sgd_iter": params['num_sgd_iter'],
            "vf_share_layers": True,
            "vf_loss_coeff": params['vf_loss_coeff'],
            "callbacks": PommeCallbacks,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: "policy_0" if agent_id % 2 == 0 else "random"),
                "policies_to_train": ["policy_0"],
            },
            "log_level": "WARN",
            "use_pytorch": True
        }
    )


if __name__ == "__main__":
    ray.shutdown()
    ray.init(object_store_memory=4e10)

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpu')
    parser.add_argument('--train_batch_size', type=int, default=65536)
    parser.add_argument('--sgd_minibatch_size', type=int, default=1024)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--num_sgd_iter', type=int, default=3)
    parser.add_argument('--vf_loss_coeff', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy_coeff', type=float, default=0.001)
    parser.add_argument('--truncate_episodes', type=bool, default=True,
                        help='if True use truncate_episodes else complete_episodes')
    parser.add_argument('--rollout_fragment_length', type=int, default=256)
    parser.add_argument('--training_iteration', type=int, default=1000)
    parser.add_argument('--checkpoint_freq', type=int, default=10)
    parser.add_argument('--num_envs_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus_per_worker', type=float, default=0.0)
    parser.add_argument('--name', type=str, default="experiment")
    parser.add_argument('--render', type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)

    training_team(params)
