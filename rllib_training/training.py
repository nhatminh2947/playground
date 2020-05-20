import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

import pommerman
from pommerman import constants
from rllib_training import models
from rllib_training.envs import pomme_env
from rllib_training.envs.pomme_env import PommeMultiAgent

tf = try_import_tf()

NUM_AGENTS = 4


# def on_episode_end(info):
#     global last_info
#     episode = info["episode"]
#
#     for agent_name in agent_names:
#         if episode.last_info_for(agent_name)["result"] != constants.Result.Incomplete:
#             last_info = episode.last_info_for(agent_name)
#             break
#     # print(episode.agent_rewards)
#     # print(episode._agent_reward_history["ppo_agent_1"])
#     # print(episode._agent_reward_history["ppo_agent_2"])
#     if "win" not in episode.custom_metrics:
#         episode.custom_metrics["win"] = 0
#     if "loss" not in episode.custom_metrics:
#         episode.custom_metrics["loss"] = 0
#     if "tie" not in episode.custom_metrics:
#         episode.custom_metrics["tie"] = 0
#     # print(last_info)
#     if last_info["result"] == constants.Result.Win:
#         if any([last_info['winners'] == [1, 3],
#                 last_info['winners'] == [1],
#                 last_info['winners'] == [3]]):
#             episode.custom_metrics["win"] += 1
#         else:
#             episode.custom_metrics["loss"] += 1
#     elif last_info["result"] == constants.Result.Tie:
#         episode.custom_metrics["tie"] += 1


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


# def on_episode_step(info):
#     episode = info["episode"]
#
#     for agent_name in agent_names:
#         action = episode.last_action_for(agent_name)
#         if action == constants.Action.Bomb.value:
#             episode.custom_metrics[agent_name + "_bomb"] += 1
#
#
# def on_episode_start(info):
#     episode = info["episode"]
#
#     for agent_name in agent_names:
#         episode.custom_metrics[agent_name + "_bomb"] = 0


def training_team():
    env_id = "PommeTeamCompetition-v0"

    env_config = {
        "env_id": "PommeTeamCompetition-v0"
    }

    env = pommerman.make(env_id, [])
    obs_space = pomme_env.DICT_SPACE_FULL
    act_space = env.action_space

    ModelCatalog.register_custom_model("1st_model", models.FirstModel)
    ModelCatalog.register_custom_model("2nd_model", models.SecondModel)
    ModelCatalog.register_custom_model("torch_conv", models.ConvNetModel)
    tune.register_env("PommeMultiAgent-v0", lambda x: PommeMultiAgent(env_config))

    # pbt = PopulationBasedTraining(
    #     time_attr="training_iteration",
    #     metric="episode_reward_mean",
    #     mode="max",
    #     perturbation_interval=10,
    #     resample_probability=0.25,
    #     hyperparam_mutations={
    #         "lambda": lambda: random.uniform(0.9, 1.0),
    #         "clip_param": lambda: random.uniform(0.01, 0.5),
    #         "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    #         "num_sgd_iter": lambda: random.randint(1, 30),
    #         "sgd_minibatch_size": lambda: random.randint(128, 16384),
    #         "train_batch_size": lambda: random.randint(2000, 160000),
    #     }
    # )

    def gen_policy():
        config = {
            "model": {
                "custom_model": "torch_conv",
                "custom_options": {
                    "in_channels": 16,
                    "feature_dim": 512
                }
            },
            "use_pytorch": True
        }
        return PPOTorchPolicy, obs_space, act_space, config

    # Setup PPO with an ensemble of `num_policies` different policies.
    policies = {
        "policy_{}".format(i): gen_policy() for i in range(2)
    }
    print(policies.keys())
    trials = tune.run(
        PPOTrainer,
        queue_trials=True,
        stop={
            "training_iteration": 10050,
        },
        checkpoint_freq=50,
        checkpoint_at_end=True,
        # scheduler=pbt,
        # num_samples=1,
        # restore="/home/nhatminh2947/ray_results/3rd_model_no_wood_static/PPO_PommeMultiAgent_9d08bc9e_0_2020-03-23_14-57-51nrucciv4/checkpoint_3500/checkpoint-3500",
        config={
            "batch_mode": "complete_episodes",
            "env": PommeMultiAgent,
            "env_config": env_config,
            "num_workers": 11,
            "num_gpus": 1,
            "train_batch_size": 50000,
            "sgd_minibatch_size": 5000,
            "clip_param": 0.2,
            "lambda": 0.995,
            "num_sgd_iter": 10,
            "vf_share_layers": True,
            "vf_loss_coeff": 1e-3,
            "callbacks": {
                # "on_train_result": on_train_result,
                # "on_episode_end": on_episode_end,
                # "on_episode_step": on_episode_step,
                # "on_episode_start": on_episode_start
            },
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: list(policies.keys())[agent_id % 2]),
                "policies_to_train": ["policy_0", "policy_1"],
            },
            # "custom_eval_function": evaluate,
            # "evaluation_interval": 1,
            # "evaluation_num_episodes": 100,
            "log_level": "WARN",
            "use_pytorch": True
        }
    )


if __name__ == "__main__":
    ray.shutdown()
    ray.init()

    training_team()
