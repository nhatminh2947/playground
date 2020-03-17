'''Entry point into the agents module set'''
from .base_agent import BaseAgent
from .docker_agent import DockerAgent
from .http_agent import HttpAgent
from .player_agent import PlayerAgent
from .player_agent_blocking import PlayerAgentBlocking
from .random_agent import RandomAgent, StaticAgent, SmartRandomAgent, SmartRandomAgentNoBomb
from .simple_agent import SimpleAgent
from .simple_agent_cautious_bomb import CautiousAgent
from .tensorforce_agent import TensorForceAgent
