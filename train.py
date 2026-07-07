import time

import numpy as np
import torch

from checkers import Checkers, Outcome
from model import ResNet, Trainer, ReplayBuffer
from mcts import MCTSNode, MCTS

device = torch.device('cpu')

canon_game = Checkers()
model = ResNet(n_blocks=3, n_channels=64, value_layers=32)
model.to(device)
replay_buffer = ReplayBuffer()
trainer = Trainer(model, device, replay_buffer)
mcts = MCTS(model, device, MCTSNode(None, Checkers(), 1.0), 100)

states = []
action_probs = []

while canon_game.outcome is None:
    start = time.time()
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
    end = time.time()
    print(f'step-time: {end - start}')
    
print(canon_game.outcome)
if canon_game.outcome == Outcome.B_WIN:
    print(f'can move: {canon_game.can_move()}')
elif canon_game.outcome == Outcome.W_WIN:
    print(f'can move: {canon_game.can_move()}')
canon_game.print_board()




    

    