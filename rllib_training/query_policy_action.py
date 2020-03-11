from ray.rllib.agents.ppo import PPOTrainer

from rllib_training.envs import PommeRllib
from pommerman import configs

config = configs.ffa_v0_fast_env()

trainer = PPOTrainer(env=PommeRllib, config={
    "num_workers": 9,
    "num_gpus": 1,
    "env_config": config,
})
