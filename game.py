from abc import ABC, abstractmethod

class Game(ABC):
    @abstractmethod
    def copy(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def get_state(self):
        pass
    
    @abstractmethod
    def get_valid_moves(self):
        pass
    