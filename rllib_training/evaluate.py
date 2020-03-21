import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

from pommerman import configs
from pommerman.agents.simple_agent import SimpleAgent
from pommerman.envs.v0 import Pomme
from rllib_training.envs.pomme_env import PommeFFA
from rllib_training.models import FirstModel, ThirdModel

ray.init()

env_config = configs.team_v0_fast_env()
env = Pomme(**env_config['env_kwargs'])
ModelCatalog.register_custom_model("third_model", ThirdModel)

ppo_agent = PPOTrainer(config={
    "env_config": configs.team_v0_fast_env(),
    "num_workers": 1,
    "model": {"custom_model": "third_model"}
}, env=PommeFFA)

pomme_env = PommeFFA()
# fdb733b6
checkpoint = 2590
checkpoint_dir = "/home/nhatminh2947/ray_results/PPO/PPO_PommeFFA_ee778b62_0_2020-03-12_20-20-59vj21bf83"
ppo_agent.restore("{}/checkpoint_{}/checkpoint-{}".format(checkpoint_dir, checkpoint, checkpoint))

agents = {}
for agent_id in range(2):
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
        actions.append(ppo_agent.compute_action(observation=pomme_env.featurize(obs[1])))
        obs, reward, done, info = env.step(actions)
        print("reward:", reward)
        print("done:", done)
        print("info:", info)
        print("=========")
    env.render(close=True)
    # env.close()
