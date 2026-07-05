from checkers import Checkers, Piece

game = Checkers()

game.turn = Piece.WHITE
game.w_score = 6
game.b_score = 7
game.board = [[Piece.EMPTY, Piece.BLACK, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.BLACK],
              [Piece.BLACK, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.BLACK, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
              [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.WHITE, Piece.EMPTY, Piece.EMPTY],
              [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
              [Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
              [Piece.BLACK, Piece.EMPTY, Piece.WHITE, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY],
              [Piece.EMPTY, Piece.WHITE, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.BLACK],
              [Piece.EMPTY, Piece.EMPTY, Piece.WHITE, Piece.EMPTY, Piece.EMPTY, Piece.EMPTY, Piece.WHITE, Piece.EMPTY]]

game.print_board()
logit = 2 * (8 * 4) + 5 * 4 + 0
print(game.get_move(logit))
game.step(logit)
game.print_board()
print(game.board)