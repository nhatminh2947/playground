from pommerman import constants
from pommerman import utility
from pommerman.envs import v0


class Pomme(v0.Pomme):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_board(self):
        self._board = utility.make_board(self._board_size, self._num_rigid,
                                         self._num_wood, len(self._agents))

        self._board[self._board == constants.Item.Wood.value] = constants.Item.Passage.value
