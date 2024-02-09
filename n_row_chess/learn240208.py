from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy



class Configuration:

    def __init__(self):
        # 游戏
        self.width = 8
        self.height = 8
        self.n_in_row = 4  # n子棋
        self.Black = 1
        self.White = -1
        self.Continue = 0
        self.Blackstone = 1
        self.Whitestone = -1
        self.Empty = 0

        # mcts
        self.searchtime = 5
        self.dirchlet = 0.3
        self.cput = 5
        self.temp = 1
        # train
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.buffer_size = 50000     # 双端队列 经验池
        self.model_path = None

class Game:
    def __init__(self, config:Configuration):
        self.width = config.width
        self.height = config.height
        self.n_in_row = config.n_in_row
        self.Black = config.Black
        self.White = config.White
        self.Blackstone = config.Blackstone
        self.Whitestone = config.Whitestone
        self.Empty = config.Empty
        self.Continue = config.Continue
        self.cur_player = self.Black
        self.board = [self.Empty for i in range(self.width * self.height)]
        self.history = []

    def xy2move(self, x, y):
        return x * self.height + y

    def move2xy(self, move):
        return move // self.height, move % self.height

    def play(self, move):
        assert self.board[move] == self.Empty, f"{move} has stone error -> play"
        if self.cur_player == self.Black:
            self.board[move] = self.Blackstone
        else:
            self.board[move] = self.Whitestone
        self.cur_player = -self.cur_player
        self.history.append(move)

        flag, winner = self.winner()
        if flag:
            return winner
        else:
            return self.Continue

    def unplay(self, move):
        assert self.board[move] != self.Empty, f"{move} has not stone error -> unplay"
        self.cur_player = -self.cur_player
        self.board[move] = self.Empty
        self.history.pop()

    def visual(self):
        '''
        (0, 6)(1, 6)(2, 6)(3, 6)(4, 6)
        (0, 5)(1, 5)(2, 5)(3, 5)(4, 5)
        (0, 4)(1, 4)(2, 4)(3, 4)(4, 4)
        (0, 3)(1, 3)(2, 3)(3, 3)(4, 3)
        (0, 2)(1, 2)(2, 2)(3, 2)(4, 2)
        (0, 1)(1, 1)(2, 1)(3, 1)(4, 1)
        (0, 0)(1, 0)(2, 0)(3, 0)(4, 0)


        6 13 20 27 34
        5 12 19 26 33
        4 11 18 25 32
        3 10 17 24 31
        2 9 16 23 30
        1 8 15 22 29
        0 7 14 21 28

        '''
        for y in reversed(range(self.height)):
            for x in range(self.width):
                move = self.xy2move(x, y)
                if self.board[move] == self.Blackstone:
                    print('B\t', end='')
                elif self.board[move] == self.Whitestone:
                    print('W\t', end='')
                    # print(f'{self.xy2move(x,y)}',end=' ')
                else:
                    print('-\t', end='')
                    # print(f'{self.xy2move(x,y)}',end=' ')
            print()
        print('*' * 50)

    def winner(self):
        for i, m in enumerate(self.history):
            player = self.Black if i % 2 == 0 else self.White
            Stone = self.Blackstone if i % 2 == 0 else self.Whitestone
            x, y = self.move2xy(m)

            # 水平
            count = 0
            for j in range(self.n_in_row):
                if x + j >= self.width:
                    count = 0
                    break
                if self.board[self.xy2move(x + j, y)] == Stone:
                    count += 1
                    if count >= self.n_in_row:
                        return True, player
                else:
                    count = 0
                    break

            # 竖直
            for j in range(self.n_in_row):
                if y + j >= self.height:
                    count = 0
                    break
                if self.board[self.xy2move(x, y + j)] == Stone:
                    count += 1
                    if count >= self.n_in_row:
                        return True, player
                else:
                    count = 0
                    break

            # y=x
            for j in range(self.n_in_row):
                if y + j >= self.height or x + j >= self.width:
                    count = 0
                    break
                if self.board[self.xy2move(x + j, y + j)] == Stone:
                    count += 1
                    if count >= self.n_in_row:
                        return True, player
                else:
                    count = 0
                    break
            # y=-x
            for j in range(self.n_in_row):
                if y - j < 0 or x + j >= self.width:
                    count = 0
                    break
                if self.board[self.xy2move(x + j, y - j)] == Stone:
                    count += 1
                    if count >= self.n_in_row:
                        return True, player
                else:
                    count = 0
                    break
        return False, -1

    def load_game(self, history):
        # self.restart()
        for m in history:
            winner = self.play(m)
            print(f'move: {m}')
            self.visual()
            if winner == self.Continue:
                continue
            elif winner == self.Black:
                print('black has win')
            else:
                print('white has win')

    def restart(self):
        self.board = [self.Empty for i in range(self.width * self.height)]
        self.cur_player = self.Black
        self.history = []

    def feature(self):
        """
        四张棋盘  黑方落子 白方落子 上一回合落子 当前落子方
        :return:
        """
        fea = torch.zeros((4, self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                move = self.xy2move(x, y)
                if self.board[move] == self.Blackstone:
                    fea[0][x][y] = self.Black
                elif self.board[move] == self.Whitestone:
                    fea[1][x][y] = self.White
        if len(self.history) > 0:
            m = self.history[-1]
            x, y = self.move2xy(m)
            fea[2][x][y] = self.board[m]
        fea[3] = self.cur_player
        return fea

    def get_legal_moves(self):
        return [i for i in range(self.width*self.height) if not i in self.history]


# 搭建残差块
class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters, )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters, )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


# 搭建骨干网络，输入：N, 4, H, W --> N, C, H, W
class Net(nn.Module):

    def __init__(self, num_channels=64, num_res_blocks=7, conf=Configuration()):
        super().__init__()
        self.conf = conf
        # 初始化特征
        self.conv_block = nn.Conv2d(in_channels=4, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_block_bn = nn.BatchNorm2d(num_channels)
        self.conv_block_act = nn.ReLU()
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(8 * conf.height * conf.width, conf.height * conf.width)
        # 价值头
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=4, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(4)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(4 * conf.height * conf.width, 64)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(64, 1)

    # 定义前向传播
    def forward(self, x):
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 8 * self.conf.height * self.conf.width])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy,dim=1)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * self.conf.height * self.conf.width])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value


# 策略值网络，用来进行模型的训练
class PolicyValueNet:

    def __init__(self, conf:Configuration):
        self.l2_const = 2e-3    # l2 正则化
        self.device = conf.device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)
        if conf.model_file:
            self.policy_value_net.load_state_dict(torch.load(conf.model_file))  # 加载模型参数

    # 输入一个批次的状态，输出一个批次的动作概率和状态价值
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    # 输入棋盘，返回每个合法动作的（动作，概率）元组列表，以及棋盘状态的分数
    def policy_value_fn(self, board):
        self.policy_value_net.eval()
        # 获取合法动作列表
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 9, 10, 9)).astype('float16')
        current_state = torch.as_tensor(current_state).to(self.device)
        # 使用神经网络进行预测
        with autocast(): #半精度fp16
            log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = log_act_probs.cpu() , value.cpu()
        act_probs = np.exp(log_act_probs.numpy().flatten()) if CONFIG['use_frame'] == 'paddle' else np.exp(log_act_probs.detach().numpy().astype('float16').flatten())
        # 只取出合法动作
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # 返回动作概率，以及状态价值
        return act_probs, value.detach().numpy()

    # 保存模型
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    # 执行一步训练
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        # 包装变量
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)
        # 清零梯度
        self.optimizer.zero_grad()
        # 设置学习率
        for params in self.optimizer.param_groups:
            # 遍历Optimizer中的每一组参数，将该组参数的学习率 * 0.9
            params['lr'] = lr
        # 前向运算
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        # 价值损失
        value_loss = F.mse_loss(input=value, target=winner_batch)
        # 策略损失
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))  # 希望两个向量方向越一致越好
        # 总的损失，注意l2惩罚已经包含在优化器内部
        loss = value_loss + policy_loss
        # 反向传播及优化
        loss.backward()
        self.optimizer.step()
        # 计算策略的熵，仅用于评估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()



class Node:
    def __init__(self, parent=None, pi=1.0, cput=1.0):
        self.parent = parent
        self.P = pi
        self.children = {}  # children[act]=child
        self.Q = 0.0
        self.N = 0.0
        self.cput = cput

    def update(self, v):
        self.Q = (self.Q * self.N + v) / (self.N + 1)
        self.N = self.N + 1

    def uct(self):
        return self.Q + self.cput * self.P * np.sqrt(self.parent.N) / (self.N + 1)

    def backup(self, v):
        if self.parent is not None:
            self.parent.backup(-v)
        self.update(v)

    def select(self):
        """
        return child's act and node with max uct
        """
        return max(self.children.items(), key=lambda act_node: act_node[1].uct())

    def expand(self, act_pi):
        """
        expand childs with (act,pi)
        """
        for (act, pi) in act_pi:
            if act not in self.children:
                self.children[act] = Node(self, pi, self.cput)

    def is_leaf(self):
        return len(self.children) == 0

    def __str__(self):
        s = f""
        for (act, child) in self.children.items():
            s += f"{act}:{child.uct}"
        return s


class MCTS:
    def __init__(self, game: Game, model, conf: Configuration):
        self.cput = conf.cput
        self.game = game
        self.root = Node(parent=None, pi=1.0, cput=self.cput)
        self.model = None
        if model is not None:
            self.model = model.to(conf.device)
        self.search_time = conf.searchtime
        self.conf = conf

    def policy(self, game, feature):

        if self.model is None:
            prob = [1 / (game.height * game.width) for i in range(game.height * game.width)]
            act_prob = [(move, prob[move]) for move in range(self.conf.width * self.conf.height)]
            q = 0
        else:
            self.model.eval()
            feature = feature.to(self.conf.device)
            p, q = self.model(feature)
            prob = np.exp(p.data.cpu().flatten().numpy()).tolist()
            for m in game.history:
                prob[m] = 0.0
            act_prob = [(move, prob[move]) for move in range(self.conf.width * self.conf.height)]
            q = q.data
        return act_prob, q

    def search(self):
        start = time.time()
        # i =0
        while time.time() - start <= self.search_time:
            self.simulate()
            # i+=1
        # print(f"i={i}")

    def simulate(self):
        node = self.root
        game = copy.deepcopy(self.game)
        # select 到 leaf
        flag = False
        winner = -1
        while True:
            if node.is_leaf():
                break
            act, node = node.select()
            winner = game.play(act)
            flag = True if winner != game.Continue else False

        # evaluate
        action_probs, q_value = self.policy(game, game.feature())

        # expand
        if not flag:
            action_probs = [(act, p) for act, p in action_probs if act not in game.history]
            node.expand(action_probs)
            if self.model is None:
                return
        else:
            if winner == -1:
                q_value = 0.0
            elif winner == game.cur_player:
                q_value = 1.0
            else:
                q_value = -1.0

        # update
        node.backup(-q_value)

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

    def get_move_probs(self, temp=1e-3):
        act_visits = []
        for act in range(self.game.width * self.game.height):
            if act in self.root.children.keys():
                act_visits.append((act, self.root.children[act].N))
            else:
                act_visits.append((act, 0.0))
        acts, visits = zip(*act_visits)
        act_probs = self.softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return list(acts), list(act_probs)

    def update_with_move(self,last_move):
        """
        重用搜索树 替代 reset
        :param last_move:
        :return:
        """
        if last_move in self.root.children.keys():
            self.root= self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(parent=None, pi=1.0, cput=self.cput)


# 自我对弈收集数据
class CollectPipeline:
    def __init__(self,init_model:Net,conf:Configuration):
        # 游戏
        self.game = Game(conf)

        self.data_buffer = deque(maxlen=conf.buffer_size)
        self.iters = 0

    def load_model(self):
        """
        取到最新的模型
        :param model_path:
        :return:
        """
