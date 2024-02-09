import json

import torch
import torch.utils.data.dataset as Dataset
from torch.utils.data import dataloader

from tensor_board import history_to_feature


class RenjuDataset(Dataset.Dataset):
    def __init__(self):
        self.datas = []
        self.labels = []

        self.games_data = self.load_file()
        for game in self.games_data[:500]:
            result = game['result']
            moves = game['moves']
            for i in range(5, (len(moves) - 1)):  # 从第三步开始起步  例如:疏星局开局

                is_black = i % 2 == 0
                v = 0
                if result == 0.5:  # 黑棋平局视作输
                    v = -1 if is_black else 1
                elif result == 1:  # 黑棋胜
                    v = 1 if is_black else -1
                else:
                    v = -1 if is_black else 1
                v = torch.tensor(v, dtype=torch.float32)

                if v == 1:
                    board = moves[:i]
                    feature = history_to_feature(board)
                    p = torch.zeros(225, dtype=torch.float32)
                    p[moves[i + 1]] = 1

                    self.datas.append(feature)
                    self.labels.append((p, v))

                    # 90度旋转
                    rotated_board_90 = torch.rot90(feature, k=1, dims=(1, 2))  # 3*15*15的张量   15*15的棋盘同时翻转
                    rotated_pi_90 = torch.rot90(p.reshape((15, 15)), k=1).flatten()
                    self.datas.append(rotated_board_90)
                    self.labels.append((rotated_pi_90, v))

                    # 180度旋转
                    rotated_board_180 = torch.rot90(feature, k=2, dims=(1, 2))
                    rotated_pi_180 = torch.rot90(p.reshape((15, 15)), k=2).flatten()
                    self.datas.append(rotated_board_180)
                    self.labels.append((rotated_pi_180, v))

                    # 270度旋转
                    rotated_board_270 = torch.rot90(feature, k=3, dims=(1, 2))
                    rotated_pi_270 = torch.rot90(p.reshape((15, 15)), k=3).flatten()
                    self.datas.append(rotated_board_270)
                    self.labels.append((rotated_pi_270, v))

    def load_file(self):
        input_filename = './data/games_data.json'
        with open(input_filename, 'r') as json_file:
            games_data = json.load(json_file)
            print(f"信息提取成功")
            return games_data

    def __len__(self):
        """
        :return: 样本长度
        """
        return len(self.datas)

    def __getitem__(self, index):
        """
        :param index:
        :return: 返回datas的一个样本  (s,p,q)
        """
        s = self.datas[index]
        p, q = self.labels[index]

        if torch.cuda.is_available():
            device = torch.device("cuda")
            s = s.to(device)
            p = p.to(device)
            q = q.to(device)
        return s, p, q

#
# renju_dataset = RenjuDataset()
#
# # input在dataloader内部 放置到cuda上
# renju_dataloader = dataloader.DataLoader(renju_dataset,
#                                          batch_size=64,
#                                          shuffle=True,
#                                          num_workers=0)  # 需注意 在windows中num_workers只能是0，在linux中才能多进程
#
# items = []
# for i, item in enumerate(renju_dataloader):
#     s, p, q = item
#     items.append(item)
#     print(item)
#
