from checkers import Checkers, Move
from model import ReplayBuffer

game = Checkers()
buffer = ReplayBuffer()
player, opponent, player_k, opponent_k, turn = game.get_state()
buffer.add((player, opponent, player_k, opponent_k, turn), (0, 0, 0, 0, 0, 0, 0, 0), 0)
while game.step():
    player, opponent, player_k, opponent_k, turn = game.get_state()
    buffer.add((player, opponent, player_k, opponent_k, turn), (0, 0, 0, 0, 0, 0, 0, 0), 0)
    print(f"current-states: \nplayer:\n{player}\nopponent:\n{opponent}\nplayer_k:\n{player_k}\nopponent_k:\n{opponent_k}\nturn:\n{turn}")
    states, action_probs, values = buffer.sample(1)
    print(f"sampled-states: turn:\n{states[0][12]}\n\nplayer:\n{states[0][0]}\nh1_player:\n{states[0][1]}\nh2_player:\n{states[0][2]}\nopponent:\n{states[0][3]}\nh1_opponent:\n{states[0][4]}\nh2_oppinent:{states[0][5]}")