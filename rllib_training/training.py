import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

import pommerman
from pommerman import constants
from rllib_training import models
from rllib_training.envs import pomme_env
from rllib_training.envs.pomme_env import PommeFFA, PommeMultiAgent

tf = try_import_tf()

# agent_names = ["simple_agent_0", "simple_agent_1", "simple_agent_2", "ppo_agent_3"]
agent_names = ["ppo_agent_1", "ppo_agent_2"]

NUM_AGENTS = 4


def on_episode_end(info):
    global last_info
    episode = info["episode"]

    for agent_name in agent_names:
        if episode.last_info_for(agent_name)["result"] != constants.Result.Incomplete:
            last_info = episode.last_info_for(agent_name)
            break
    # print(episode.agent_rewards)
    # print(episode._agent_reward_history["ppo_agent_1"])
    # print(episode._agent_reward_history["ppo_agent_2"])
    if "win" not in episode.custom_metrics:
        episode.custom_metrics["win"] = 0
    if "loss" not in episode.custom_metrics:
        episode.custom_metrics["loss"] = 0
    if "tie" not in episode.custom_metrics:
        episode.custom_metrics["tie"] = 0
    # print(last_info)
    if last_info["result"] == constants.Result.Win:
        episode.custom_metrics["win"] += 1
    elif last_info["result"] == constants.Result.Tie:
        episode.custom_metrics["tie"] += 1
    else:
        episode.custom_metrics["loss"] += 1


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

    for agent_name in agent_names:
        action = episode.last_action_for(agent_name)
        if action == constants.Action.Bomb.value:
            episode.custom_metrics[agent_name + "_bomb"] += 1


def on_episode_start(info):
    episode = info["episode"]

    for agent_name in agent_names:
        episode.custom_metrics[agent_name + "_bomb"] = 0


def on_postprocess_traj(info):
    # curiosity_rews = info["pre_batch"][0].model.curiosity_loss
    # info["post_batch"][1]['rewards'] += curiosity_rews
    print(info["post_batch"])


def training_team():
    env_id = "PommeTeam-nowood-v0"

    env_config = {
        "agent_names": agent_names,
        "env_id": env_id,
        "phase": 0
    }

    env = pommerman.make(env_id, [])
    obs_space = pomme_env.DICT_SPACE_FULL
    act_space = env.action_space

    ModelCatalog.register_custom_model("1st_model", models.FirstModel)
    ModelCatalog.register_custom_model("2nd_model", models.SecondModel)
    ModelCatalog.register_custom_model("3rd_model", models.ThirdModel)
    ModelCatalog.register_custom_model("rnd_model", models.RNDModel)

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

    trials = tune.run(
        PPOTrainer,
        name="3rd_model_no_wood_static",
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
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "num_gpus": 1,
            "train_batch_size": 50000,
            "sgd_minibatch_size": 5000,
            "clip_param": 0.2,
            "lambda": 0.95,
            "num_sgd_iter": 10,
            "vf_share_layers": True,
            "vf_loss_coeff": 1e-3,
            "callbacks": {
                "on_train_result": on_train_result,
                "on_episode_end": on_episode_end,
                "on_episode_step": on_episode_step,
                "on_episode_start": on_episode_start,
                "on_postprocess_traj": on_postprocess_traj
            },
            "multiagent": {
                "policies": {
                    "ppo_policy": (PPOTFPolicy, obs_space, act_space, {
                        "model": {
                            "custom_model": "rnd_model",
                            # "use_lstm": True,
                            # "max_seq_len": 10,
                        }
                    }),
                },
                "policy_mapping_fn": (lambda agent_id: "ppo_policy"),
                "policies_to_train": ["ppo_policy"],
            },
            # "custom_eval_function": evaluate,
            # "evaluation_interval": 1,
            # "evaluation_num_episodes": 100,
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


def evaluate(trainer, eval_workers):
    """Example of a custom evaluation function.
    Arguments:
        trainer (Trainer): trainer class to evaluate.
        eval_workers (WorkerSet): evaluation workers.
    Returns:
        metrics (dict): evaluation metrics dict.
    """

    # We configured 2 eval workers in the training config.
    worker_1, worker_2 = eval_workers.remote_workers()

    # Set different env settings for each worker. Here we use a fixed config,
    # which also could have been computed in each worker by looking at
    # env_config.worker_index (printed in SimpleCorridor class above).
    worker_1.foreach_env.remote(lambda env: env.set_corridor_length(4))
    worker_2.foreach_env.remote(lambda env: env.set_corridor_length(7))

    for i in range(5):
        print("Custom evaluation round", i)
        # Calling .sample() runs exactly one episode per worker due to how the
        # eval workers are configured.
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])

    # Collect the accumulated episodes on the workers, and then summarize the
    # episode stats into a metrics dict.
    episodes, _ = collect_episodes(remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
    # You can compute metrics from the episodes manually, or use the
    # convenient `summarize_episodes()` utility:
    metrics = summarize_episodes(episodes)
    # Note that the above two statements are the equivalent of:
    # metrics = collect_metrics(eval_workers.local_worker(),
    #                           eval_workers.remote_workers())

    # You can also put custom values in the metrics dict.
    metrics["foo"] = 1
    return metrics


if __name__ == "__main__":
    ray.shutdown()
    ray.init(num_gpus=1)

    training_team()
