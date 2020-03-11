import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

from rllib_training.envs.pma import PommeFFA
from rllib_training.models.first_model import FirstModel

tf = try_import_tf()

# agent_names = ["simple_agent_0", "simple_agent_1", "simple_agent_2", "ppo_agent_3"]
agent_names = ["ppo_agent_0", "ppo_agent_1", "ppo_agent_2", "ppo_agent_3"]

NUM_AGENTS = 4

if __name__ == "__main__":
    ray.shutdown()
    ray.init(local_mode=False)

    ModelCatalog.register_custom_model("first_model", FirstModel)

    tune.run(
        PPOTrainer,
        stop={
            "training_iteration": 10000,
        },
        checkpoint_freq=10,
        checkpoint_at_end=True,
        config={
            "env": PommeFFA,
            "model": {
                "custom_model": "first_model"
            },
            "num_workers": 10,
            "num_gpus": 1,
            "log_level": "WARN",
        }
    )
