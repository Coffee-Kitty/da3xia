"""

评测   nn+500mcts  500mcts  1000mcts   2000mcts
的elo吧
    轮流先手对战500局相互   即  24*500局共
"""
from game import Board, Game
from mcts_alphazero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_pure_player
from policy_value_net import PolicyValueNet


def compute_Ex(rank1, rank2):
    """
    返回 （1的预期胜率,2的）
    >>> compute_Ex(0,0)
    (0.5, 0.5)
    """
    e1 = 1 / (1 + pow(10, (rank2 - rank1) / 400))
    return e1, 1 - e1


def update_rank(rank_old, k, Ea, Sa):
    return rank_old + k * (Sa - Ea)


if __name__ == "__main__":
    board = Board(width=8, height=8, n_in_row=4)
    game = Game(board, False)
    policy_value_net = PolicyValueNet(8,
                                      8,
                                      model_file='model/best_policy.model')
    nn_mcts500 = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=500, is_selfplay=0)
    mcts500 = MCTS_pure_player(5, 500)
    mcts1000 = MCTS_pure_player(5, 1000)
    mcts2000 = MCTS_pure_player(5, 2000)
    playes = [nn_mcts500, mcts500, mcts1000, mcts2000]
    elos = [0.5 for i in range(len(playes))]
    ranks = [0 for i in range(len(playes))]
    k = 20
    games = 500
    while games:
        games -= 1
        print(f"games: {500-games}轮")

        for i in range(len(playes) - 1):
            for j in range(i + 1, len(playes)):
                player1 = playes[i]
                player2 = playes[j]

                for forehand in [0, 1]:
                    win = game.start_play(player1, player2, forehand)
                    ea, eb = compute_Ex(ranks[i], ranks[j])
                    # print(f"i: {i}")
                    # print(f"j: {j}")
                    # print(f"ea: {ea}")
                    # print(f"eb: {eb}")
                    if win == 1:
                        Sa = 1
                    elif win == 2:
                        Sa = 0
                    else:
                        Sa = 0.5
                    ranks[i] = update_rank(ranks[i], k, ea, Sa)
                    ranks[j] = update_rank(ranks[j], k, eb, 1 - Sa)

        print("ranks:{}".format(ranks))
