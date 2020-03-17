import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.models import ModelCatalog

import pommerman
from pommerman import agents
from pommerman import configs
from pommerman.envs.v0 import Pomme
from rllib_training.envs import pomme_env
from rllib_training.models.first_model import FirstModel

ray.init()

env_config = configs.phase_0_team_v0_env()
env = Pomme(**env_config['env_kwargs'])
obs_space = pomme_env.DICT_SPACE_FULL
act_space = env.action_space
ModelCatalog.register_custom_model("first_model", FirstModel)
agent_names = ["ppo_agent_1", "ppo_agent_2"]

ppo_agent = PPOTrainer(config={
    "env_config": {
        "agent_names": agent_names,
        "env_id": "Phase0-PommeTeam-v0"
    },
    "num_workers": 1,
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
    }
}, env=pomme_env.PommeMultiAgent)

# fdb733b6
checkpoint = 20
checkpoint_dir = "/home/nhatminh2947/ray_results/PPO/PPO_PommeMultiAgent_77266e84_0_2020-03-17_05-16-56l5u2nqv0"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agents_list = [agents.StaticAgent(),
               agents.StaticAgent(),
               agents.StaticAgent(),
               agents.StaticAgent()]

env = pommerman.make("Phase0-PommeTeam-v0", agents_list)

penv = pomme_env.PommeMultiAgent({
    "agent_names": agent_names,
    "env_id": "Phase0-PommeTeam-v0"
})

for i in range(1):
    obs = env.reset()

    done = False
    while not done:
        env.render()
        actions = env.act(obs)

        actions[1] = ppo_agent.compute_action(observation=penv.featurize(obs[1]), policy_id="ppo_policy")
        actions[3] = ppo_agent.compute_action(observation=penv.featurize(obs[3]), policy_id="ppo_policy")

        obs, reward, done, info = env.step(actions)
        print("actions:", actions)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)
        print("=========")
    env.render(close=True)
    # env.close()
