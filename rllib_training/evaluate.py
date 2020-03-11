import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog

from pommerman.agents.simple_agent import SimpleAgent
from pommerman.configs import ffa_competition_env
from pommerman.envs.v0 import Pomme
from rllib_training.envs.pma import PommeFFA
from rllib_training.models.first_model import FirstModel

# config = ffa_v0_fast_env()
# env = Pomme(**config["env_kwargs"])
#
# agents = {}
# for agent_id in range(3):
#     agents[agent_id] = SimpleAgent(config["agent"](agent_id, config["game_type"]))
# env.set_agents(list(agents.values()))
# env.set_init_game_state(None)
#
# env.seed(0)
# obs = env.reset()
#
# # Run the random agents until we're done
# done = False
# while not done:
#     env.render()
#     actions = env.act(obs)ModuleNotFoundError: No module named 'training.envs'; 'training' is not a package
#     obs, reward, done, info = env.step(actions)
# env.render(close=True)
# env.close()
ray.init()
# config['model']['fcnet_hiddens'] = [100, 100]
env_config = ffa_competition_env()
env = Pomme(**env_config['env_kwargs'])
ModelCatalog.register_custom_model("first_model", FirstModel)

ppo_agent = PPOTrainer(config={
    "num_workers": 1,
    "model": {"custom_model": "first_model"}
}, env=PommeFFA)

pomme_env = PommeFFA()

checkpoint = 170
ppo_agent.restore(
    "/home/nhatminh2947/ray_results/PPO/PPO_PommeFFA_7dddb7f4_0_2020-03-11_23-32-18yp0c0f7b/checkpoint_{}/checkpoint-{}".format(
        checkpoint, checkpoint))

agents = {}
for agent_id in range(4):
    agents[agent_id] = SimpleAgent(env_config["agent"](agent_id, env_config["game_type"]))
env.set_agents(list(agents.values()))
env.set_init_game_state(None)
env.training_agent = 3

obs = env.reset()

done = False
while not done:
    env.render()
    actions = env.act(obs)
    actions.append(ppo_agent.compute_action(observation=pomme_env.featurize(obs[3])))
    obs, reward, done, info = env.step(actions)
    print("reward:", reward)
    print("done:", done)
    print("info:", info)
    print("=========")
env.render(close=True)
env.close()
