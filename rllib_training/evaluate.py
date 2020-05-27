import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

from pommerman import configs
from pommerman.agents.simple_agent import SimpleAgent
from pommerman.envs.v0 import Pomme
from rllib_training import models
from rllib_training.envs import PommeMultiAgent

ray.init()
env_id = "PommeTeam-v0"

env_config = configs.team_v0_env()
env = Pomme(**env_config['env_kwargs'])
ModelCatalog.register_custom_model("torch_conv", models.ActorCriticModel)


def gen_policy():
    config = {
        "model": {
            "custom_model": "torch_conv",
            "custom_options": {
                "in_channels": 16,
                "feature_dim": 512
            }
        },
        "gamma": 0.999,
        "use_pytorch": True
    }
    return PPOTorchPolicy, obs_space, act_space, config


policies = {
    "policy_{}".format(i): gen_policy() for i in range(2)
}

ppo_agent = PPOTrainer(config={
    "env_config": {
        "env_id": env_id,
        "render": True
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["ppo_policy"],
    },
    "num_workers": 0,
    "model": {"custom_model": "torch_conv"}
}, env=PommeMultiAgent)

# fdb733b6
checkpoint = 240
checkpoint_dir = "/home/nhatminh2947/ray_results//home/lucius/ray_results/experiment/PPO_PommeMultiAgent_0_2020-05-27_20-34-17w_avvlrt"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agents = {}
for agent_id in range(4):
    agents[agent_id] = SimpleAgent(env_config["agent"](agent_id, env_config["game_type"]))
env.set_agents(list(agents.values()))
env.set_init_game_state(None)
env.training_agent = 1

for i in range(1):
    obs = env.reset()

    done = False
    while not done:
        env.render()
        actions = env.act(obs)
        actions[0] = ppo_agent.compute_action(observation=PommeMultiAgent.featurize(obs[0]))
        actions[2] = ppo_agent.compute_action(observation=PommeMultiAgent.featurize(obs[2]))
        obs, reward, done, info = env.step(actions)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)
        print("=========")
    env.render(close=True)
    # env.close()
