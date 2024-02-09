import torch

# 棋盘上位置的状态
EMPTY = -1
WHITE = 0
BLACK = 1
# 游戏胜利与否的状态
CONTINUE = 2
NO_CLEAR_WINNER = 3  # 平局
# 棋盘大小
BOARD_LEN = 15
# 神经网络输入1*3*15*15的特征
FEATURE_LEN = 3


def history_to_feature(history):
    """
        返回向神经网络输入的特征 3*15*15
        3*15*15
        第一张15*15的棋盘是我方有棋子处为1，其余位置都为0
        第二张15*15的棋盘是对方有棋子处为1，其余位置都为0
        第三张15*15的棋盘是如果我方是黑方，全为-1，我方是白方，全为1
    """
    board = [EMPTY for i in range(BOARD_LEN * BOARD_LEN)]

    first_player = BLACK  # 我方  即 在当前回合下 该落子的一方
    for act in history:
        board[act] = first_player
        first_player = BLACK + WHITE - first_player

    feature = torch.zeros((FEATURE_LEN, BOARD_LEN, BOARD_LEN),dtype=torch.float32)

    for i in range(BOARD_LEN):
        for j in range(BOARD_LEN):
            pos = i * BOARD_LEN + j
            if board[pos] == first_player:
                feature[0][i][j] = 1
            elif board[pos] == WHITE + BLACK - first_player:
                feature[1][i][j] = 1
    if first_player == BLACK:
        feature[2] = -1
    elif first_player == WHITE:
        feature[2] = 1

    return feature

# his = [112, 113, 111, 110, 128, 96, 82, 127]
# feature = history_to_feature(his)
# print(feature)
