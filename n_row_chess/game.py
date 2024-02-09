
import numpy as np
import import_ipynb
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
from collections import deque, defaultdict
import random
from torch.autograd import Variable




class Configuration:

    def __init__(self):
        # 游戏
        self.width = 7
        self.height = 7
        self.n_in_row = 4  # n子棋
        self.Black = 1
        self.White = -1
        self.Continue = 0
        self.Blackstone = 1
        self.Whitestone = -1
        self.Empty = 0

        # train
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.learn_rate = 2e-4 # 基准学习率
        self.lr_multiplier = 1.0  # 基于KL自动调整学习倍速
        self.temp = 1.0  # 温度参数
        self.search_time = 5  # 每下一步棋，搜索时间
        self.c_puct = 5 # exploitation和exploration之间的折中系数
        self.buffer_size = 10000
        self.batch_size = 256 # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size) #使用 deque 创建一个双端队列
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02 # 早停检查
        self.check_freq = 50# 每50次检查一次，策略价值网络是否更新
        self.game_batch_epoch= 500 # 训练多少个epoch
        self.best_win_ratio = 0.0 # 当前最佳胜率，用他来判断是否有更好的模型
        # 弱AI（纯MCTS）模拟步数，用于给训练的策略AI提供对手
        self.pure_mcts_playout_num = 1000
        self.init_model = 'best_policy.model'



class Game:
    def __init__(self,config = Configuration()):
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
        self.board = [self.Empty for i in range(self.width*self.height)]
        self.history = []

    def xy2move(self,x, y):
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
                    print('B\t',end='')
                elif self.board[move] == self.Whitestone:
                    print('W\t',end='')
                    # print(f'{self.xy2move(x,y)}',end=' ')
                else:
                    print('-\t', end='')
                    # print(f'{self.xy2move(x,y)}',end=' ')
            print()
        print('*'*50)

    def winner(self):
        for i, m in enumerate(self.history):
            player = self.Black if i % 2==0 else self.White
            Stone = self.Blackstone if i%2 ==0 else self.Whitestone
            x , y = self.move2xy(m)

            # 水平
            count = 0
            for j in range(self.n_in_row):
                if x+j >= self.width:
                    count = 0
                    break
                if self.board[self.xy2move(x+j ,y)] == Stone:
                    count += 1
                    if count >= self.n_in_row:
                        return True, player
                else:
                    count = 0
                    break

            # 竖直
            for j in range(self.n_in_row):
                if y+j >= self.height:
                    count = 0
                    break
                if self.board[self.xy2move(x ,y+j)] == Stone:
                    count += 1
                    if count >= self.n_in_row:
                        return True, player
                else:
                    count = 0
                    break

            # y=x
            for j in range(self.n_in_row):
                if y+j >= self.height or x+j >=self.width:
                    count = 0
                    break
                if self.board[self.xy2move(x+j ,y+j)] == Stone:
                    count += 1
                    if count >= self.n_in_row:
                        return True, player
                else:
                    count = 0
                    break
            # y=-x
            for j in range(self.n_in_row):
                if y-j < 0 or x+j >=self.width:
                    count = 0
                    break
                if self.board[self.xy2move(x+j ,y-j)] == Stone:
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
         self.board = [self.Empty for i in range(self.width*self.height)]
         self.cur_player = self.Black
         self.history = []

    def feature(self):
        """
        四张棋盘  黑方落子 白方落子 上一回合落子 当前落子方
        :return:
        """
        fea = torch.zeros((4, self.width,self.height))
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




class Net(nn.Module):

    def __init__(self,conf:Configuration):
        super(Net, self).__init__()
        self.board_width = conf.width
        self.board_height = conf.height
        # 通用层 common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 行动策略层 action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*self.board_width*self.board_height,
                                 self.board_width*self.board_height)
        # 状态值层 state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*self.board_width*self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, feature):
         # 通用层 common layers
        x = F.relu(self.conv1(feature))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行动策略层 action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act),dim=1)
        # 状态值层 state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        # 输出行动可能性 和 终局的预期状态值
        return x_act, x_val
# conf = Configuration()
# net = Net(conf)
# input = torch.zeros((2,4,conf.width,conf.height))
# acts,q = net(input)
# print(acts.shape)
# print(q.shape)


# In[ ]:


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
        return self.Q + self.cput * self.P * np.sqrt(self.parent.N ) / (self.N + 1)

    def backup(self, v):
        if self.parent is not None:
            self.parent.backup(-v)
        self.update(v)

    def select(self):
        """
        return child's act and node with max uct
        """
        return max(self.children.items(),key=lambda act_node:act_node[1].uct())

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
    def __init__(self, game:Game, cput, model, conf:Configuration,search_time=5):
        self.cput = cput
        self.game = game
        self.root = Node(parent=None, pi=1.0, cput=cput)
        self.model=None
        if model is not None:
            self.model = model.to(conf.device)
        self.search_time = search_time
        self.conf = conf

    def reset(self):
        self.root = Node(parent=None, pi=1.0, cput=self.cput)

    def policy(self,game, feature):

        if self.model is None:
            prob = [1/(game.height*game.width) for i in range(game.height*game.width)]
            act_prob = [(move,prob[move]) for move in range(self.conf.width*self.conf.height)]
            q = 0
        else:
            self.model.eval()
            feature = feature.to(self.conf.device)
            p,q = self.model(feature)
            prob = np.exp(p.data.cpu().flatten().numpy()).tolist()
            for m in game.history:
                prob[m] = 0.0
            act_prob = [(move,prob[move]) for move in range(self.conf.width*self.conf.height)]
            q = q.data
        return act_prob,q

    def search(self):
        start = time.time()
        # i =0
        while time.time()-start <= self.search_time:
            self.simulate()
            # i+=1
        # print(f"i={i}")

    def simulate(self):
        node = self.root
        game = copy.deepcopy(self.game)
        # select 到 leaf
        flag= False
        winner = -1
        while True:
            if node.is_leaf():
                break
            act,node = node.select()
            winner = game.play(act)
            flag = True if winner != game.Continue else False

        # evaluate
        action_probs, q_value = self.policy(game,game.feature())

        # expand
        if not flag:
            action_probs = [(act,p) for act,p in action_probs if act not in game.history]
            node.expand(action_probs)
            if self.model is None:
                return
        else:
            if winner == -1:
                q_value = 0.0
            elif winner==game.cur_player:
                q_value = 1.0
            else:
                q_value = -1.0

        # update
        node.backup(-q_value)

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

    def get_move_probs(self,temp=1e-3):
        act_visits=[]
        for act in range(self.game.width*self.game.height):
            if act in self.root.children.keys():
                act_visits.append((act, self.root.children[act].N))
            else:
                act_visits.append((act, 0.0))
        acts, visits = zip(*act_visits)
        act_probs = self.softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return list(acts), list(act_probs)


# conf = Configuration()
# net=Net(conf)
# game = Game()
# game.load_game([0,7,1,9,2,5])
# mcts = MCTS(game,1.0,None,conf,search_time=10)
# mcts.search()
# acts, act_probs = mcts.get_move_probs()
# move = acts[act_probs.index(max(act_probs))]
# print(move)


# In[1]:

class Train:
    def __init__(self,conf:Configuration):
        self.conf = conf
        self.device = conf.device
        self.learn_rate = conf.learn_rate # 基准学习率
        self.lr_multiplier = conf.lr_multiplier # 基于KL自动调整学习倍速
        self.temp = conf.temp # 温度参数
        self.search_time = conf.search_time # 每下一步棋，搜索时间
        self.c_puct = conf.c_puct # exploitation和exploration之间的折中系数
        self.buffer_size = conf.buffer_size
        self.batch_size = conf.batch_size # mini-batch size for training
        self.data_buffer =conf.data_buffer #使用 deque 创建一个双端队列

        self.play_batch_size = conf.play_batch_size
        self.epochs = conf.epochs  # num of train_steps for each update
        self.kl_targ = conf.kl_targ # 早停检查
        self.check_freq = conf.check_freq # 每50次检查一次，策略价值网络是否更新

        self.game_batch_epoch= conf.game_batch_epoch # 训练多少个epoch
        self.best_win_ratio = conf.best_win_ratio# 当前最佳胜率，用他来判断是否有更好的模型
        # 弱AI（纯MCTS）模拟步数，用于给训练的策略AI提供对手
        self.pure_mcts_playout_num = conf.pure_mcts_playout_num
        self.init_model = conf.init_model
        self.net = Net(conf)
        if self.init_model is not None:
            net_parm = torch.load(self.init_model)
            self.net.load_state_dict(net_parm)


        self.l2_const = 1e-4  # L2正则项系数
        self.optimizer = optim.Adam(self.net.parameters(), weight_decay=self.l2_const)

    # 通过旋转和翻转增加数据集, play_data: [(state, mcts_prob, winner_z), ..., ...]
    def get_equi_data(self, play_data):
        extend_data = []
        for state, mcts_porb, winner in play_data:
            state = np.array(state)
            mcts_porb = np.array(mcts_porb)
            winner = np.array(winner)
            # 在4个方向上进行expand，每个方向都进行旋转，水平翻转
            for i in [1,2,3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(mcts_porb.reshape(self.conf.width, self.conf.height)), i)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(), winner))

        return extend_data

    def collect_selfplay_data(self, n_games=1):
        game = Game(self.conf)
        mcts = MCTS(game,1.0,self.net,self.conf,search_time=self.search_time)
        playdata =[]
        for i in range(n_games):
             # 记录该局对应的数据：states, mcts_probs, current_players
             states, mcts_probs, current_players = [], [], []
             game.restart()
             while True:
                mcts.reset()
                mcts.search()
                acts, act_probs = mcts.get_move_probs(temp=self.temp)
                for m in game.history:
                    act_probs[m]=0.0
                # history
                mcts_probs.append(act_probs)
                states.append(game.feature().cpu().numpy().tolist())
                current_players.append(game.cur_player)

                move = acts[act_probs.index(max(act_probs))]
                winner = game.play(move)
                # game.visual()
                end = True if winner !=game.Continue or len(game.history) == game.width*game.height else False

                if end:
                    # end
                    winners_z = np.zeros(len(current_players))
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                    if winner == game.Black:
                        print('black has win')
                    else:
                        print('white has win')

                    playdata.extend(list(zip(states, mcts_probs, current_players)))
                    break
        # 保存下了多少步
        self.episode_len = len(playdata)
        playdata = self.get_equi_data(playdata)
        self.data_buffer.extend(playdata)

    def train_step(self,state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    lr):

        # 清空模型中参数的梯度，即梯度置为0
        self.net.train()
        self.net.zero_grad()
        # 设置学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # 前向传播
        log_act_probs, value = self.net(state_batch)
        # 定义 loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch*log_act_probs, 1))
        loss = value_loss + policy_loss
        # 反向传播，优化参数
        loss.backward()
        self.optimizer.step()
        # 计算Policy信息熵
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        # 返回loss和entropy
        return loss.item(), entropy.item()
     # 更新策略网络
    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        # 保存更新前的old_probs, old_v
        state_batch = torch.from_numpy(np.array(state_batch,dtype=np.float32)).to(torch.device("cuda"))
        mcts_probs_batch = torch.from_numpy(np.array(mcts_probs_batch,dtype=np.float32)).to(torch.device("cuda"))
        winner_batch = torch.from_numpy(np.array(winner_batch,dtype=np.float32)).to(torch.device("cuda"))

        old_probs, old_v = self.net(state_batch)
        old_probs, old_v = old_probs.detach().cpu().numpy(), old_v.detach().cpu().numpy()
        for i in range(self.epochs):
            # 每次训练，调整参数，返回loss和entropy
            loss, entropy = self.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            # 输入状态，得到行动的可能性和状态值，按照batch进行输入
            new_probs, new_v = self.net(state_batch)
            new_probs, new_v =new_probs.detach().cpu().numpy(), new_v.detach().cpu().numpy()
            # 计算更新前后两次的loss差
            epsilon = 1e-10
            kl = np.mean(np.sum(old_probs * (
                    np.log(np.clip(old_probs+ 1e-10, epsilon, 1.0 - epsilon)) - np.log(np.clip(new_probs+ 1e-10, epsilon, 1.0 - epsilon))),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # 动态调整学习倍率 lr_multiplier
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        mcts_probs_batch, winner_batch = mcts_probs_batch.detach().cpu().numpy(), winner_batch.detach().cpu().numpy()
        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    # 用于评估训练网络的质量，评估一共10场play，返回比赛胜率（赢1分、输0分、平0.5分）
    def policy_evaluate(self, n_games=10):
        game = Game()
        current_mcts_player = MCTS(game,self.c_puct,self.net,self.conf,search_time=3)
        pure_mcts_player = MCTS(game,self.c_puct,None,self.conf,search_time=3)

        mywin = 0

        for i in range(n_games):
           game.restart()
           if i < n_games//2:
               forehand=False
               print("pure mcts 先手")
           else:
               forehand=True
               print("pure mcts 后手")

           while True:
                current_mcts_player.reset()
                pure_mcts_player.reset()
                if forehand:
                    current_mcts_player.search()
                    acts, act_probs=current_mcts_player.get_move_probs(self.temp)
                    for m in game.history:
                        act_probs[m] = 0.0
                    move = acts[act_probs.index(max(act_probs))]
                    winner = game.play(move)
                    print(f"nn move:{move}")
                else:
                    pure_mcts_player.search()
                    acts, act_probs=pure_mcts_player.get_move_probs(self.temp)
                    for m in game.history:
                        act_probs[m] = 0.0
                    move = acts[act_probs.index(max(act_probs))]
                    winner = game.play(move)
                    print(f"pure mcts move:{move}")
                forehand = not forehand
                game.visual()
                if winner!=game.Continue or len(game.history) >= game.width*game.height:
                    break


           if winner == game.White and i < n_games//2:
               mywin += 1
           elif winner == game.Black and i >= n_games//2:
               mywin += 1
        # 计算胜率，平手计为0.5分
        win_ratio = mywin*1.0 / n_games
        print(" win_ratio: {}".format(win_ratio))
        return win_ratio


    def run(self):
        try:
             # 训练game_batch_num次，每个batch比赛play_batch_size场
            for i in range(self.game_batch_epoch):
                # 收集自我对弈数据
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i+1, self.episode_len))
                # train
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # 判断当前模型的表现，保存最优模型
                if (i+1) % self.check_freq == 0: #50次檢查一次
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    # 保存当前策略
                    torch.save(self.net.state_dict(),'./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("发现新的最优策略，进行策略更新")
                        self.best_win_ratio = win_ratio
                        # 更新最优策略
                        torch.save(self.net.state_dict(), './best_policy.model')
                        # if self.best_win_ratio == 1.0 and self.search_time < 10:
                        #     self.search_time += 1
                        #     print(f"searchtime : {self.search_time}")
                        #     self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ =='__main__':
    conf = Configuration()
    train = Train(conf)
    train.run()

    # conf = Configuration()
    # game = Game()
    # game.load_game([45, 1, 23, 8, 32, 15])
    # net = Net(conf)
    # if conf.init_model is not None:
    #     net_parm = torch.load(conf.init_model)
    #     net.load_state_dict(net_parm)
    # mcts = MCTS(game, 1.0, net, conf, search_time=5)
    # while True:
    #     mcts.reset()
    #     mcts.search()
    #     acts, act_probs = mcts.get_move_probs()
    #     print(act_probs)
    #     move = acts[act_probs.index(max(act_probs))]
    #     print(game.history)
    #     winner = game.play(move)
    #     if winner != game.Continue:
    #         break
    #     game.visual()
    #
    #     x, y = int(input('x= \t')), int(input('y= \t'))
    #     move = game.xy2move(x, y)
    #     winner = game.play(move)
    #     if winner != game.Continue:
    #         break
    #     print(game.history)
    #     game.visual()

