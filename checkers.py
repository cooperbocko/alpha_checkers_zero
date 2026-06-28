from enum import Enum

class Piece(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    K_BLACK = 3
    K_WHITE = 4

class Move(Enum):
    UPLEFT = 0
    UPRIGHT = 1
    DOWNLEFT = 2
    DOWNRIGHT = 3
    
class Outcome(Enum):
    W_WIN = 1
    DRAW = 0
    B_WIN = -1

#Note: American Checkers Rules
class Checkers:
    def __init__(self):
        self.outcome = None
        self.turn = Piece.BLACK
        self.prev_jump = None
        self.w_score = 0
        self.b_score = 0
        self.draw_moves = 0
        self.board = [[Piece.EMPTY for _ in range(8)] for _ in range(8)]
        for i in range(3):
            for j in range(8):
                if i % 2 == 0 and j % 2 == 1:
                    self.board[i][j] = Piece.WHITE
                elif i % 2 == 1 and j % 2 == 0:
                    self.board[i][j] = Piece.WHITE
        for i in range(5, 8):
            for j in range(8):
                if i % 2 == 0 and j % 2 == 1:
                    self.board[i][j] = Piece.BLACK
                elif i % 2 == 1 and j % 2 == 0:
                    self.board[i][j] = Piece.BLACK
                    
    def step(self):
        if self.check_winner():
            print("Game Over")
            return False
        if self.check_draw():
            print("Game Over")
            return False
        
        self.print_board()
        while True:
            print("Enter position: row column")
            position = tuple(map(int, input().split()))
            print("Enter move: 0 = up left, 1 = up right, 2 = down left, 3 = down right")
            move = Move(int(input()))
            if not self.move(position, move):
                print("Invalid move!")
            else:
                break
        
        return True
    
    def get_state(self):
        pass
    
    #Moves: 8x8x4 order -> up left, up right, down left, down right
    def get_valid_moves(self) -> list[int]:
        moves = [0 for _ in range(8 * 8 * 4)]
        for i in range(8):
            for j in range(8):
                if self.check_valid_move((i, j), Move.UPLEFT, self.can_jump(), self.is_jump((i, j), Move.UPLEFT)):
                    moves[i * 8 * 4 + j * 4 + 0] = 1
                if self.check_valid_move((i, j), Move.UPRIGHT, self.can_jump(), self.is_jump((i, j), Move.UPRIGHT)):
                    moves[i * 8 * 4 + j * 4 + 1] = 1
                if self.check_valid_move((i, j), Move.DOWNLEFT, self.can_jump(), self.is_jump((i, j), Move.DOWNLEFT)):
                    moves[i * 8 * 4 + j * 4 + 2] = 1
                if self.check_valid_move((i, j), Move.DOWNRIGHT, self.can_jump(), self.is_jump((i, j), Move.DOWNRIGHT)):
                    moves[i * 8 * 4+ j * 4 + 3] = 1
        return moves
    
    def get_move(self, logit: int) -> tuple[tuple[int, int], Move]:
        row = logit // (8 * 4)
        col = (logit - row * 8 * 4) // 4
        move = logit - row * 8 * 4 - col * 4
        return ((row, col), Move(move))
    
    def print_board(self):
        print("  0 1 2 3 4 5 6 7")
        for i in range(8):
            print(i, end = " ")
            for j in range(8):
                if self.board[i][j] == Piece.BLACK:
                    print("b", end = " ")
                elif self.board[i][j] == Piece.WHITE:
                    print("w", end = " ")
                elif self.board[i][j] == Piece.K_BLACK:
                    print("B", end = " ")
                elif self.board[i][i] == Piece.K_WHITE:
                    print("W", end = " ")
                else:
                    print("-", end = " ")
            print()
        print(f"B: {self.b_score} W: {self.w_score} Turn: {'W' if self.turn == Piece.WHITE else 'B'}")
            
    def move(self, position: tuple[int, int], move: Move):
        jump = self.can_jump()
        is_jump = self.is_jump(position, move)
        if not self.check_valid_move(position, move, jump, is_jump):
            return False
        
        piece = self.board[position[0]][position[1]]
        self.board[position[0]][position[1]] = Piece.EMPTY
        if is_jump:
            if move == Move.UPLEFT:
                taken_piece = self.board[position[0] - 1][position[1] - 1]
                self.board[position[0] - 1][position[1] - 1] = Piece.EMPTY
                self.board[position[0] - 2][position[1] - 2] = piece
                new_position = (position[0] - 2, position[1] - 2)
            elif move == Move.UPRIGHT:
                taken_piece = self.board[position[0] - 1][position[1] + 1]
                self.board[position[0] - 1][position[1] + 1] = Piece.EMPTY
                self.board[position[0] - 2][position[1] + 2] = piece
                new_position = (position[0] - 2, position[1] + 2)
            elif move == Move.DOWNLEFT:
                taken_piece = self.board[position[0] + 1][position[1] - 1]
                self.board[position[0] + 1][position[1] - 1] = Piece.EMPTY
                self.board[position[0] + 2][position[1] - 2] = piece
                new_position = (position[0] + 2, position[1] - 2)
            elif move == Move.DOWNRIGHT:
                taken_piece = self.board[position[0] + 1][position[1] + 1]
                self.board[position[0] + 1][position[1] + 1] = Piece.EMPTY
                self.board[position[0] + 2][position[1] + 2] = piece
                new_position = (position[0] + 2, position[1] + 2)
            
            if taken_piece == Piece.BLACK or taken_piece == Piece.K_BLACK:
                self.w_score += 1
            elif taken_piece == Piece.WHITE or taken_piece == Piece.K_WHITE:
                self.b_score += 1
                
            if self.is_jump(new_position, Move.UPLEFT) or self.is_jump(new_position, Move.UPRIGHT) or self.is_jump(new_position, Move.DOWNLEFT) or self.is_jump(new_position, Move.DOWNRIGHT):
                self.prev_jump = new_position
            else:
                self.prev_jump = None
                self.turn = Piece.WHITE if self.turn == Piece.BLACK else Piece.BLACK
        else:
            if move == Move.UPLEFT:
                self.board[position[0] - 1][position[1] - 1] = piece
                new_position = (position[0] - 1, position[1] - 1)
            elif move == Move.UPRIGHT:
                self.board[position[0] - 1][position[1] + 1] = piece
                new_position = (position[0] - 1, position[1] + 1)
            elif move == Move.DOWNLEFT:
                self.board[position[0] + 1][position[1] - 1] = piece
                new_position = (position[0] + 1, position[1] - 1)
            elif move == Move.DOWNRIGHT:
                self.board[position[0] + 1][position[1] + 1] = piece
                new_position = (position[0] + 1, position[1] + 1)
            self.turn = Piece.WHITE if self.turn == Piece.BLACK else Piece.BLACK
        
        if piece == Piece.BLACK and new_position[0] == 0:
            self.board[new_position[0]][new_position[1]] = Piece.K_BLACK
        elif piece == Piece.WHITE and new_position[0] == 7:
            self.board[new_position[0]][new_position[1]] = Piece.K_WHITE
            
        if piece == Piece.BLACK or piece == Piece.WHITE or is_jump:
            self.draw_moves = 0
        else:
            self.draw_moves += 1
        
        self.move_count += 1
        return True
            
    def check_valid_move(self, position: tuple[int, int], move: Move, jump: bool, is_jump: bool):
        #No move for empty space
        piece = self.board[position[0]][position[1]]
        if piece == Piece.EMPTY:
            return False
        
        #Can only move your pieces
        if (piece == Piece.WHITE or piece == Piece.K_WHITE) and self.turn != Piece.WHITE:
            return False
        if (piece == Piece.BLACK or piece == Piece.K_BLACK) and self.turn != Piece.BLACK:
            return False
        
        #Jump chain
        if self.prev_jump: 
            if self.prev_jump != position:
                return False
            
        #Must take jump
        if jump and not is_jump:
            return False
        elif jump and is_jump:
            return True
        
        #Regular moves
        if piece == Piece.BLACK:
            if move == Move.DOWNLEFT or move == Move.DOWNRIGHT:
                return False
            elif move == Move.UPLEFT:
                if position[0] - 1 < 0 or position[1] - 1 < 0:
                    return False
                if self.board[position[0] - 1][position[1] - 1] != Piece.EMPTY:
                    return False
            elif move == Move.UPRIGHT:
                if position[0] - 1 < 0 or position[1] + 1 >= 8:
                    return False
                if self.board[position[0] - 1][position[1] + 1] != Piece.EMPTY:
                    return False
        elif piece == Piece.WHITE:
            if move == Move.UPLEFT or move == Move.UPRIGHT:
                return False
            elif move == Move.DOWNLEFT:
                if position[0] + 1 >= 8 or position[1] - 1 < 0:
                    return False
                if self.board[position[0] + 1][position[1] - 1] != Piece.EMPTY:
                    return False
            elif move == Move.DOWNRIGHT:
                if position[0] + 1 >= 8 or position[1] + 1 >= 8:
                    return False
                if self.board[position[0] + 1][position[1] + 1] != Piece.EMPTY:
                    return False
        elif piece == Piece.K_BLACK or piece == Piece.K_WHITE:
            if move == Move.UPLEFT:
                if position[0] - 1 < 0 or position[1] - 1 < 0:
                    return False
                if self.board[position[0] - 1][position[1] - 1] != Piece.EMPTY:
                    return False
            elif move == Move.UPRIGHT:
                if position[0] - 1 < 0 or position[1] + 1 >= 8:
                    return False
                if self.board[position[0] - 1][position[1] + 1] != Piece.EMPTY:
                    return False
            elif move == Move.DOWNLEFT:
                if position[0] + 1 >= 8 or position[1] - 1 < 0:
                    return False
                if self.board[position[0] + 1][position[1] - 1] != Piece.EMPTY:
                    return False
            elif move == Move.DOWNRIGHT:
                if position[0] + 1 >= 8 or position[1] + 1 >= 8:
                    return False
                if self.board[position[0] + 1][position[1] + 1] != Piece.EMPTY:
                    return False
        return True
    
    def is_jump(self, position: tuple[int, int], move: Move):
        piece = self.board[position[0]][position[1]]
        if piece == Piece.EMPTY:
            return False
        i = position[0]
        j = position[1]
        
        up_left = (i - 1, j - 1) if i - 1 >= 0 and j - 1 >= 0 else None
        up_left_2 = (i - 2, j - 2) if i - 2 >= 0 and j - 2 >= 0 else None
        up_right = (i - 1, j + 1) if i - 1 >= 0 and j + 1 < 8 else None
        up_right_2 = (i - 2, j + 2) if i - 2 >= 0 and j + 2 < 8 else None
        down_left = (i + 1, j - 1) if i + 1 < 8 and j - 1 >= 0 else None
        down_left_2 = (i + 2, j - 2) if i + 2 < 8 and j - 2 >= 0 else None
        down_right = (i + 1, j + 1) if i + 1 < 8 and j + 1 < 8 else None
        down_right_2 = (i + 2, j + 2) if i + 2 < 8 and j + 2 < 8 else None
        
        if piece == Piece.BLACK:
            if move == Move.UPLEFT:
                if up_left and self.board[up_left[0]][up_left[1]] == Piece.WHITE and up_left_2 and self.board[up_left_2[0]][up_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.UPRIGHT:
                if up_right and self.board[up_right[0]][up_right[1]] == Piece.WHITE and up_right_2 and self.board[up_right_2[0]][up_right_2[1]] == Piece.EMPTY:
                    return True
        elif piece == Piece.WHITE:
            if move == Move.DOWNLEFT:
                if down_left and self.board[down_left[0]][down_left[1]] == Piece.BLACK and down_left_2 and self.board[down_left_2[0]][down_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNRIGHT:
                if down_right and self.board[down_right[0]][down_right[1]] == Piece.BLACK and down_right_2 and self.board[down_right_2[0]][down_right_2[1]] == Piece.EMPTY:
                    return True
        elif piece == Piece.K_BLACK:
            if move == Move.UPLEFT:
                if up_left and self.board[up_left[0]][up_left[1]] == Piece.WHITE and up_left_2 and self.board[up_left_2[0]][up_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.UPRIGHT:
                if up_right and self.board[up_right[0]][up_right[1]] == Piece.WHITE and up_right_2 and self.board[up_right_2[0]][up_right_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNLEFT:
                if down_left and self.board[down_left[0]][down_left[1]] == Piece.WHITE and down_left_2 and self.board[down_left_2[0]][down_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNRIGHT:
                if down_right and self.board[down_right[0]][down_right[1]] == Piece.WHITE and down_right_2 and self.board[down_right_2[0]][down_right_2[1]] == Piece.EMPTY:
                    return True
        if piece == Piece.K_WHITE:
            if move == Move.UPLEFT:
                if up_left and self.board[up_left[0]][up_left[1]] == Piece.BLACK and up_left_2 and self.board[up_left_2[0]][up_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.UPRIGHT:
                if up_right and self.board[up_right[0]][up_right[1]] == Piece.BLACK and up_right_2 and self.board[up_right_2[0]][up_right_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNLEFT:
                if down_left and self.board[down_left[0]][down_left[1]] == Piece.BLACK and down_left_2 and self.board[down_left_2[0]][down_left_2[1]] == Piece.EMPTY:
                    return True
            elif move == Move.DOWNRIGHT:
                if down_right and self.board[down_right[0]][down_right[1]] == Piece.BLACK and down_right_2 and self.board[down_right_2[0]][down_right_2[1]] == Piece.EMPTY:
                    return True
        return False
    
    def can_jump(self):
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece == Piece.EMPTY:
                    continue
                
                if self.turn == Piece.BLACK:
                    if piece == Piece.WHITE or piece == Piece.K_WHITE:
                        continue
                if self.turn == Piece.WHITE:
                    if piece == Piece.BLACK or piece == Piece.K_BLACK:
                        continue
                
                if self.is_jump((i, j), Move.UPLEFT) or self.is_jump((i, j), Move.UPRIGHT) or self.is_jump((i, j), Move.DOWNLEFT) or self.is_jump((i, j), Move.DOWNRIGHT):
                    return True
        return False
    
    def check_winner(self):
        if self.b_score == 12:
            print("Black wins!")
            self.outcome = Outcome.B_WIN
            return True
        elif self.w_score == 12:
            print("White wins!")
            self.outcome = Outcome.W_WIN
            return True
        elif not self.can_move():
            if self.turn == Piece.BLACK:
                print("White wins!")
                self.outcome = Outcome.W_WIN
            else:
                print("Black wins!")
                self.outcome = Outcome.B_WIN
            return True
        return False
    
    def check_draw(self):
        if self.move_count == self.max_moves:
            print("Draw!")
            self.outcome = Outcome.DRAW
            return True
        return False
            
    def can_move(self):
        for i in range(8):
            for j in range(8):
                piece = self.board[i][j]
                if piece == Piece.EMPTY:
                    continue
                
                #Jumps
                if self.can_jump():
                    return True      
                
                #Regular moves
                if self.turn == Piece.BLACK and piece == Piece.WHITE or piece == Piece.K_WHITE:
                    continue
                if self.turn == Piece.WHITE and piece == Piece.BLACK or piece == Piece.K_BLACK:
                    continue
                if self.check_valid_move((i, j), Move.DOWNLEFT, False, False) or self.check_valid_move((i, j), Move.DOWNRIGHT, False, False) or self.check_valid_move((i, j), Move.UPLEFT, False, False) or self.check_valid_move((i, j), Move.UPRIGHT, False, False):
                        return True
        return False
                
check = Checkers()
while True:
    if not check.step():
        break
        