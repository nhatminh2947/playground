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

tf = try_import_tf()

NUM_AGENTS = 4


def on_episode_end(info):
    global last_info
    episode = info["episode"]

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


def on_train_result(info):
    result = info["result"]

    if "phase" not in result.keys():
        result["phase"] = 0

    if result["phase"] == 0 and result["episode_len_mean"] >= 400:
        print("Next phase")
        result["phase"] += 1
        result["phase"] = min(result["phase"], 2)

        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(result["phase"])))
    elif result["phase"] == 1 and result["episode_reward_mean"] >= 2:
        print("Next phase")
        result["phase"] += 1
        result["phase"] = min(result["phase"], 2)

        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(result["phase"])))
    elif result["phase"] == 2 and result["episode_reward_mean"] >= 1:
        print("Next phase")
        result["phase"] += 1
        result["phase"] = min(result["phase"], 2)

        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_phase(result["phase"])))


def on_episode_step(info):
    episode = info["episode"]

    for agent_name in range(4):
        action = episode.last_action_for(agent_name)
        if action == constants.Action.Bomb.value:
            episode.custom_metrics["bomb_agent_{}".format(agent_name)] += 1


def on_episode_start(info):
    episode = info["episode"]

    for agent_name in range(4):
        episode.custom_metrics["bomb_agent_{}".format(agent_name)] = 0


def training_team():
    env_id = "PommeTeamCompetition-v0"

    env_config = {
        "env_id": "PommeTeamCompetition-v0"
    }

    env = pommerman.make(env_id, [])
    obs_space = spaces.Box(low=0, high=20, shape=(16, 11, 11))
    act_space = env.action_space

    ModelCatalog.register_custom_model("1st_model", models.FirstModel)
    ModelCatalog.register_custom_model("2nd_model", models.SecondModel)
    ModelCatalog.register_custom_model("torch_conv", models.ConvNetModel)
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
            "gamma": 0.995,
            "use_pytorch": True
        }
        return PPOTorchPolicy, obs_space, act_space, config

    policies = {
        "policy_{}".format(i): gen_policy() for i in range(2)
    }
    print(policies.keys())

    trials = tune.run(
        PPOTrainer,
        queue_trials=True,
        stop={
            "training_iteration": 1000,
            # "timesteps_total": 100000000
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        config={
            "lr": 1e-4,
            "entropy_coeff": 0.01,
            "kl_coeff": 0.0,
            "batch_mode": "truncate_episodes",
            "env": PommeMultiAgent,
            "env_config": env_config,
            "num_workers": 8,
            "num_envs_per_worker": 16,
            "num_gpus": 1,
            "train_batch_size": 65536,
            "sgd_minibatch_size": 1024,
            "clip_param": 0.2,
            "lambda": 0.95,
            "num_sgd_iter": 6,
            "vf_share_layers": True,
            "vf_loss_coeff": 0.5,
            "callbacks": {
                # "on_train_result": on_train_result,
                "on_episode_end": on_episode_end,
                "on_episode_step": on_episode_step,
                "on_episode_start": on_episode_start
            },
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: list(policies.keys())[agent_id % 2]),
                "policies_to_train": ["policy_0", "policy_1"],
            },
            "log_level": "WARN",
            "use_pytorch": True
        }
    )


if __name__ == "__main__":
    ray.shutdown()
    ray.init(object_store_memory=4e10)

    training_team()
