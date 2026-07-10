import time
import multiprocessing
import os

import numpy as np
import torch

from checkers import Checkers, Outcome
from model import ResNet, Trainer, ReplayBuffer
from mcts import MCTSNode, MCTS

def data_worker(id, data_queue, stop_event, iterations):
    print(f'worker {id} started')
    
    device = torch.device('cpu')
    model = ResNet(n_blocks=5, n_channels=64, value_layers=32)
    if os.path.exists('temp_checkpoint.pt'):
        model.load_state_dict(torch.load('temp_checkpoint.pt'))
    model.to(device)
    model.eval()
    
    while not stop_event.is_set():
        start = time.time()
        mcts = MCTS(model, device, MCTSNode(None, Checkers(), 1.0), iterations)
        canon_game = Checkers()
        action_probs = []
        states = []
        
        while canon_game.outcome is None and not stop_event.is_set():
            state = canon_game.get_state()
            probs = np.zeros(256)
            action, valid_target_probs = mcts.get_move()
            for v_action, prob in valid_target_probs.items():
                probs[v_action] = prob
            states.append(state)
            action_probs.append(probs)
            canon_game.step(action)

        outcome = canon_game.outcome
        if outcome == Outcome.DRAW:
            rewards = np.zeros(len(states))
        else:
            rewards = np.zeros(len(states))
            for i, state in enumerate(states):
                if state[4][0][0] == outcome:
                    rewards[i] = 1
                else:
                    rewards[i] = -1
        
        if stop_event.is_set():
            return
        data_queue.put((states, action_probs, rewards))
        end = time.time()
        print(f'worker {id} finished a game in {end - start} seconds')
    
    print(f'worker {id} stopped')

if __name__ == "__main__":
    n_workers = 3
    iterations = 100
    gpu_device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    base_model = ResNet(n_blocks=5, n_channels=64, value_layers=32)
    base_model.to(gpu_device)
    replay_buffer = ReplayBuffer()
    trainer = Trainer(base_model, gpu_device, replay_buffer)
    
    with multiprocessing.Manager() as manager:
        data_queue = manager.Queue()
        stop_event = manager.Event()
    
        workers = []
        for i in range(n_workers):
            w = multiprocessing.Process(target=data_worker, args=(i, data_queue, stop_event, iterations))
            w.start()
            workers.append(w)
        
        while True:
            while not data_queue.empty():
                states, action_probs, rewards = data_queue.get_nowait()
                replay_buffer.add(states, action_probs, rewards)
                
            if len(replay_buffer.buffer) > 0:
                print(f'training!')
                trainer.train(32)
            time.sleep(1)
            



    

    