from enum import Enum

import numpy as np

from game import Game

class Outcome(int, Enum):
    DRAW = 0
    X_WIN = 1
    O_WIN = 2
    
class Piece(int, Enum):
    EMPTY = 0
    X = 1
    O = 2

class TicTacToe(Game):
    def __init__(self):
        self.turn = Piece.X
        self.outcome = None
        self.board = np.zeros((3, 3), dtype=np.int8)
        
    def copy(self):
        temp = TicTacToe()
        temp.turn = self.turn
        temp.outcome = self.outcome
        temp.board = np.copy(self.board)
        return temp
    
    def step(self, action: int):
        self.move(self.get_move(action))
        if self.check_win:
            return False
        return True
    
    def move(self, position):
        self.board[position[0]][position[1]] = self.turn
        self.turn = Piece.O if self.turn == Piece.X else Piece.X
    
    def get_state(self):
        if self.turn == Piece.X:
            player = (self.board == Piece.X).astype(np.float32)
            opponent = (self.board == Piece.O).astype(np.float32)
            turn = np.zeros((3, 3))
        else:
            player = (self.board == Piece.O).astype(np.float32)
            opponent = (self.board == Piece.X).astype(np.float32)
            turn = np.ones((3, 3))
        return np.stack((player, opponent, turn))
    
    def get_valid_moves(self):
        moves = np.zeros((9), dtype=bool)
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Piece.EMPTY:
                    moves[i] = 1
        return moves
    
    def get_move(self, logit: int):
        row = logit // 3
        col = logit - row * 3
        return (row, col)
    
    def print_board(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == Piece.X:
                    print("X", end = " ")
                elif self.board[i][j] == Piece.O:
                    print("O", end = " ")
                else:
                    print("-", end = " ")
            print()
    
    def check_win(self, postion):
        piece = self.board[postion[0]][postion[1]]
        up = self.board[postion[0] - 1][postion[1]] if postion[0] > 0 else None
        down = self.board[postion[0] + 1][postion[1]] if postion[0] < 2 else None
        left = self.board[postion[0]][postion[1] - 1] if postion[1] > 0 else None
        right = self.board[postion[0]][postion[1] + 1] if postion[1] < 2 else None
        up_left_diag = self.board[postion[0] - 1][postion[1] - 1] if postion[0] > 0 and postion[1] > 0 else None
        up_right_diag = self.board[postion[0] - 1][postion[1] + 1] if postion[0] > 0 and postion[1] < 2 else None
        down_left_diag = self.board[postion[0] + 1][postion[1] - 1] if postion[0] < 2 and postion[1] > 0 else None
        down_right_diag = self.board[postion[0] + 1][postion[1] + 1] if postion[0] < 2 and postion[1] < 2 else None
        
        if (up == piece and down == piece) or (left == piece and right == piece) or (up_left_diag == piece and down_right_diag == piece) or (up_right_diag == piece and down_left_diag == piece):
            self.outcome = Outcome.X_WIN if piece == Piece.X else Outcome.O_WIN
            return True
        
        return False