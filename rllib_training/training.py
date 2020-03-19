import random

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import Trainable
from pommerman import configs
from pommerman.envs.v0 import Pomme
from rllib_training.envs import pomme_env
from rllib_training.envs.pomme_env import PommeFFA, PommeMultiAgent
from rllib_training.models.first_model import FirstModel

tf = try_import_tf()

# agent_names = ["simple_agent_0", "simple_agent_1", "simple_agent_2", "ppo_agent_3"]
agent_names = ["ppo_agent_0", "ppo_agent_1", "ppo_agent_2", "ppo_agent_3"]

NUM_AGENTS = 4


def training_team():
    env_config = configs.team_v0_env()
    env = Pomme(**env_config['env_kwargs'])
    obs_space = pomme_env.DICT_SPACE
    act_space = env.action_space

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=10,
        resample_probability=0.25,
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            "sgd_minibatch_size": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        }
    )

    tune.run(
        PPOTrainer,
        stop={
            "training_iteration": 1,
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        # scheduler=pbt,
        num_samples=1,
        restore="/home/nhatminh2947/ray_results/PPO/PPO_PommeMultiAgent_19121a0a_2_2020-03-12_04-50-54xeczcy3d/checkpoint_680/checkpoint-680",
        config={
            "env": PommeMultiAgent,
            "env_config": env_config,
            "num_workers": 4,
            "num_gpus": 1,
            "lambda": 0.95,
            "clip_param": 0.2,
            "lr": 1e-5,
            "multiagent": {
                "policies": {
                    "ppo_policy": (PPOTFPolicy, obs_space, act_space, {
                        "model": {
                            "custom_model": "first_model"
                        }
                    }),
                },
                "policy_mapping_fn": (lambda agent_id: "ppo_policy"),
                "policies_to_train": ["ppo_policy"],
            },
            "log_level": "WARN",
        }
    )


def training_ffa(env_conf):
    tune.run(
        PPOTrainer,
        stop={
            "training_iteration": 10000,
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        config={
            "env": PommeFFA,
            "env_config": env_conf,
            "model": {
                "custom_model": "first_model"
            },
            "num_workers": 10,
            "num_gpus": 1,
            "log_level": "WARN",
        }
    )


class PommeTrainer(Trainable):

    def _train(self):
        pass

    def _save(self, tmp_checkpoint_dir):
        pass

    def _restore(self, checkpoint):
        pass


if __name__ == "__main__":
    ray.shutdown()
    ray.init(local_mode=False)

    ModelCatalog.register_custom_model("first_model", FirstModel)

    training_ffa(configs.one_vs_one_env())
