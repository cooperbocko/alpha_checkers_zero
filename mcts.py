import math
import numpy as np

from checkers import Checkers
from model import ResNet

class MCTSNode:
    def __init__(self, parent, state: Checkers, prior_prob: float):
        self.parent = parent
        self.state = state
        self.children = {}
        self.visits = 0
        self.t_action_value = 0
        self.m_action_value = 0
        self.prior_prob = prior_prob
        
    def is_leaf(self):
        return len(self.children) == 0
        
class MCTS:
    def __init__(self, model: ResNet, root: MCTSNode, iterations: int, temperature: float = 1.0):
        self.model = model
        self.root = root
        self.temperature = temperature
        self.iterations = iterations
        
    def get_move(self):
        self.search()
            
        actions = self.root.children.keys()
        visits = np.array([child.visits for child in self.root.children])
        total_visits = np.sum(visits)
        visits = visits ** (1 / self.temperature)
        probs = visits / total_visits
        
        action = np.random.choice(actions, p=probs)
        action_probs = {actions[i]: probs[i] for i in range(len(actions))}
        self.root = self.root.children[action]
        self.root.parent = None
        
        return action, action_probs
        
    def search(self):
        for _ in range(self.iterations):
            # select
            node = self.select(self.root)
            
            # expand
            action_probs, value = self.model(node.state.get_state())
            self.expand(node, action_probs)
            
            # backprop
            self.backpropagate(node, value)
        
    def select(self, node: MCTSNode):
        while not node.is_leaf():
            argmax = float('-inf')
            best_node = None
            
            for action, child in node.children.items():
                q = child.m_action_value
                if child.state.turn != node.state.turn:
                    q *= -1
                    
                u = 1.5 * child.prior_prob * (math.sqrt(node.visits) / (1 + child.visits))
                value = q + u
                
                if value > argmax:
                    argmax = value
                    best_node = child
                
            node = best_node
        return node
        
    def expand(self, node: MCTSNode, action_probs):
        for i, action in node.state.get_valid_moves().enumerate():
            if action == 1:
                new_state = node.state.copy()
                position, move = new_state.get_move(i)
                new_state.move(position, move)
                child = MCTSNode(node, new_state, action_probs[i])
                node.children[i] = child
        
    def backpropagate(self, node: MCTSNode, value):
        while node is not self.root:
            node.visits += 1
            if self.root.state.turn == node.state.turn:
                node.t_action_value += value
            else:
                node.t_action_value -= value
            node.m_action_value = node.t_action_value / node.visits
            node = node.parent