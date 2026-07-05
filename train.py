import numpy as np

import time

from checkers import Checkers, Outcome
from model import ResNet, Trainer, ReplayBuffer
from mcts import MCTSNode, MCTS

canon_game = Checkers()
model = ResNet(n_blocks=3, n_channels=64, value_layers=32)
replay_buffer = ReplayBuffer()
trainer = Trainer(model, 'cpu', replay_buffer)
mcts = MCTS(model, MCTSNode(None, Checkers(), 1.0), 5)

states = []
action_probs = []
start = time.time()
while canon_game.outcome is None:
    canon_game.print_board()
    state = canon_game.get_state()
    probs = np.zeros(256)
    action, valid_target_probs = mcts.get_move()
    for v_action, prob in valid_target_probs.items():
        probs[v_action] = prob
    states.append(state)
    action_probs.append(probs)
    canon_game.step(action)
    print(f'ACTION: {canon_game.get_move(action)}')
    
print(canon_game.outcome)
if canon_game.outcome == Outcome.B_WIN:
    print(f'can move: {canon_game.can_move()}')
elif canon_game.outcome == Outcome.W_WIN:
    print(f'can move: {canon_game.can_move()}')
canon_game.print_board()

end = time.time()
print(f'time: {end - start}')


    

    