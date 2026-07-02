from checkers import Checkers, Move
from model import ReplayBuffer

game = Checkers()
buffer = ReplayBuffer()
states = []

# inital state
player, opponent, player_k, opponent_k, turn = game.get_state()
game.print_board()
print(f"current-states: \nplayer:\n{player}\nopponent:\n{opponent}\nplayer_k:\n{player_k}\nopponent_k:\n{opponent_k}\nturn:\n{turn}")
states.append((player, opponent, player_k, opponent_k, turn))

print('MOVE 1 --------')
game.move((5, 0), Move.UPRIGHT)
player, opponent, player_k, opponent_k, turn = game.get_state()
game.print_board()
print(f"current-states: \nplayer:\n{player}\nopponent:\n{opponent}\nplayer_k:\n{player_k}\nopponent_k:\n{opponent_k}\nturn:\n{turn}")
states.append((player, opponent, player_k, opponent_k, turn))

print('MOVE 2 --------')
game.move((2, 1), Move.DOWNLEFT)
player, opponent, player_k, opponent_k, turn = game.get_state()
game.print_board()
print(f"current-states: \nplayer:\n{player}\nopponent:\n{opponent}\nplayer_k:\n{player_k}\nopponent_k:\n{opponent_k}\nturn:\n{turn}")
states.append((player, opponent, player_k, opponent_k, turn))

print('MOVE 3 --------')
game.move((5, 6), Move.UPRIGHT)
player, opponent, player_k, opponent_k, turn = game.get_state()
game.print_board()
print(f"current-states: \nplayer:\n{player}\nopponent:\n{opponent}\nplayer_k:\n{player_k}\nopponent_k:\n{opponent_k}\nturn:\n{turn}")
states.append((player, opponent, player_k, opponent_k, turn))

print('SAMPLE --------')
buffer.add(states, (0, 0, 0, 0), (0, 0, 0, 0))
states, action_probs, values = buffer.sample(1)
print(f"sampled-states: turn:\n{states[0][12]}\n\nplayer:\n{states[0][0]}\nh1_player:\n{states[0][1]}\nh2_player:\n{states[0][2]}\nopponent:\n{states[0][3]}\nh1_opponent:\n{states[0][4]}\nh2_oppinent:\n{states[0][5]}")