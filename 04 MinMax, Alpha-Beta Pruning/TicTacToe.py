from minmax import *
from alphabeta import *

# game = 'oxoxNxNoN'

game = 'oxoxNNxoN'

gamestate = np.array(list(game)).reshape((3, 3))

# Game = alphaBetaNode(gamestate)
Game = minMaxNode(gamestate)
Game.fillWithChildren()
# Game.maxValue(-np.Inf, np.Inf)
# Game.maxValue(-1, 1)
Game.maxValue()
