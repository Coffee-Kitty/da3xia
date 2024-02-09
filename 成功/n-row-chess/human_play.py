from game import Game, Board
from mcts_alphazero import MCTSPlayer, MCTS
from policy_value_net import PolicyValueNet


class Human:

    def __init__(self, width, height):
        self.agent = 'HUMAN'
        self.width = width
        self.height = height

    def get_action(self, move):
        x, y =int(str(input('x= '))), int(str(input('y= ')))
        move = x*self.width+y
        return move

    def set_player_ind(self, p):
        self.player = p



if __name__ =="__main__":
    board = Board()
    game = Game(board,True)
    policy_value_net = PolicyValueNet(8,
                                      8,
                                      model_file='model/best_policy.model')
    mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,5,2000,0)
    human_player = Human(8,8)
    game.start_play(human_player,mcts_player,1)

    # board = Board()
    # board.init_board(1)
    # board.do_move(2)
    # policy_value_net = PolicyValueNet(8,
    #                                   8,
    #                                   model_file='model/best_policy.model')
    # mcts = MCTS(policy_value_net.policy_value_fn,5,2000)
    # acts, act_probs = mcts.get_move_probs(board)
    # win_rate=mcts.get_win_rate()
    # board.visual()
    # print(act_probs)
    # print(list(zip(acts,act_probs)))
    # print(win_rate)
