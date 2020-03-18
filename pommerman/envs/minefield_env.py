import numpy as np

from pommerman import constants, characters
from pommerman import utility
from pommerman.envs import v0


class Pomme(v0.Pomme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bomb_rate = 0.3

    def make_board(self):
        self._board = utility.make_board(self._board_size, self._num_rigid,
                                         self._num_wood, len(self._agents))

        self._board[self._board == constants.Item.Wood.value] = constants.Item.Wood.Passage.value
        self._board[self._board == constants.Item.Agent0.value] = constants.Item.Wood.Passage.value
        self._board[self._board == constants.Item.Agent2.value] = constants.Item.Wood.Passage.value

    def reset(self):
        assert (self._agents is not None)

        if self._init_game_state is not None:
            self.set_json_info()
        else:
            self._step_count = 0
            self.make_board()
            self.make_items()
            self._bombs = []
            self._flames = []
            self._powerups = []
            for agent_id, agent in enumerate(self._agents):
                if agent_id % 2 == 0:
                    agent.set_start_position((0, 0))
                    agent.reset(is_alive=False)
                else:
                    pos = np.where(self._board == utility.agent_value(agent_id))
                    row = pos[0][0]
                    col = pos[1][0]
                    agent.set_start_position((row, col))
                    agent.reset()

        return self.get_observations()

    def step(self, actions):
        self._intended_actions = actions

        max_blast_strength = self._agent_view_size or 10
        result = self.model.step(
            actions,
            self._board,
            self._agents,
            self._bombs,
            self._items,
            self._flames,
            max_blast_strength=max_blast_strength)
        self._board, self._agents, self._bombs, self._items, self._flames = \
            result[:5]

        if np.random.choice([0, 1], p=[1 - self.bomb_rate, self.bomb_rate]):
            while True:
                x = np.random.choice(11)
                y = np.random.choice(11)

                if utility.position_is_passage(self._board, (x, y)):
                    self._board[(x, y)] = constants.Item.Bomb.value
                    self._bombs.append(characters.Bomb(bomber=characters.Bomber(),
                                                       position=(x, y),
                                                       life=np.random.choice(np.arange(5, 10)) + 1,
                                                       blast_strength=np.random.choice(np.arange(2, 6))))
                    break

        done = self._get_done()
        obs = self.get_observations()
        reward = self._get_rewards()
        info = self._get_info(done, reward)

        if done:
            # Callback to let the agents know that the game has ended.
            for agent in self._agents:
                agent.episode_end(reward[agent.agent_id])

        self._step_count += 1
        return obs, reward, done, info

    def _get_done(self):
        alive = [agent for agent in self._agents if agent.is_alive]

        if self._step_count >= self._max_steps:
            return True

        return len(alive) <= 0

    def _get_rewards(self):
        if self._step_count >= self._max_steps:
            return [1 if agent.is_alive else -1 for agent in self._agents]

        return [0] * 4
