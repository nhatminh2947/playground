import json
import os

from pommerman.agents import BaseAgent, SimpleAgent, PlayerAgent
import pommerman
from pommerman import utility, constants

NUM_AGENTS = 4


class MctsNode(object):
    def __init__(self, parent=None):
        self.q = 0
        self.n = 0
        self.parent = parent
        pass


class MctsAgent(BaseAgent):
    def __init__(self, *args, **kwargs):

        self.step_count = 0
        self.env = self.make_env()
        self.env.set_training_agent(0)
        self.state = {
            "agents": [{"agent_id": 0, "is_alive": True, "position": [2, 1], "ammo": 1, "blast_strength": 2,
                        "can_kick": False},
                       {"agent_id": 1, "is_alive": True, "position": [2, 1], "ammo": 1, "blast_strength": 2,
                        "can_kick": False},
                       {"agent_id": 2, "is_alive": True, "position": [2, 1], "ammo": 1, "blast_strength": 2,
                        "can_kick": False},
                       {"agent_id": 3, "is_alive": True, "position": [2, 1], "ammo": 1, "blast_strength": 2,
                        "can_kick": False}],
            'board': [],
            'board_size': constants.BOARD_SIZE,
            'flames': [],
            'intended_actions': [0, 0, 0, 0],
            'items': [],
            'step_count': 0}

        """The expected game_state_file JSON format is:
          - agents: list of agents serialized (agent_id, is_alive, position,
            ammo, blast_strength, can_kick)
          - board: board matrix topology (board_size^2)
          - board_size: board size
          - bombs: list of bombs serialized (position, bomber_id, life,
            blast_strength, moving_direction)
          - flames: list of flames serialized (position, life)
          - items: list of item by position
          - step_count: step count"""
        super().__init__(*args, **kwargs)

    def update_state(self, obs):
        board = obs['board']
        bomb_life = obs['bomb_life']
        blast_strength = obs['bomb_blast_strength']
        moving_direction = obs['bomb_moving_direction']
        flame_life = obs['flame_life']
        alive = obs['alive']

        prev_state = self.state.copy()

        agents = []
        bombs = []
        flames = []
        items = []

        self.state['board'] = board.tolist().copy()

        if self.state['step_count'] != 0:
            for bomb in prev_state['bombs']:
                bomb['life'] -= 1
                if bomb['life'] != 0:
                    bombs.append(bomb)

        for agent in range(constants.Item.Agent0.value, constants.Item.Agent0.value + 4):
            self.state['agents'][agent - constants.Item.Agent0.value]['is_alive'] = agent in alive

        for row in range(constants.BOARD_SIZE):
            for col in range(constants.BOARD_SIZE):
                if utility.position_is_agent(board, (row, col)):
                    agent_id = board[row, col] - constants.Item.Agent0.value
                    self.state['agents'][agent_id]['position'] = [row, col]

                    if self.state['step_count'] != 0:
                        if prev_state['board'][row][col] == constants.Item.Kick.value:
                            self.state['agents'][agent_id]['can_kick'] = True
                        elif prev_state['board'][row][col] == constants.Item.ExtraBomb.value:
                            self.state['agents'][agent_id]['ammo'] += 1
                        elif prev_state['board'][row][col] == constants.Item.IncrRange.value:
                            self.state['agents'][agent_id]['blast_strength'] += 1

                if bomb_life[row, col] == float(constants.DEFAULT_BOMB_LIFE):
                    agent_id = board[row, col] - constants.Item.Agent0.value
                    bombs.append({"position": [row, col],
                                  "bomber_id": int(agent_id),
                                  "life": constants.DEFAULT_BOMB_LIFE,
                                  "blast_strength": blast_strength[row, col],
                                  "moving_direction": moving_direction[row, col]})

                if flame_life[row, col] != 0:
                    flames.append({'position': [row, col],
                                   'life': flame_life[row, col].tolist()})

        self.state['bombs'] = bombs
        self.state['step_count'] += 1
        self.state['flames'] = flames
        print(self.state)
        with open('./../records/mcts_states.json', 'w') as f:
            f.write(json.dumps(self.state, sort_keys=True, indent=4))

    def make_env(self):
        agents = [PlayerAgent()]
        for agent_id in range(NUM_AGENTS - 1):
            agents.append(SimpleAgent())

        return pommerman.make('PommeFFACompetition-v0', agents)

    def select(self):
        return

    def act(self, obs, action_space):
        self.update_state(obs)
        return action_space.sample()

    def episode_end(self, reward):
        pass
