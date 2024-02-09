import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import dataloader

from model import RenjuModel
from renju_dataset import RenjuDataset
from tensor_board import history_to_feature


class TrainNet:
    def __init__(self, model_path):
        self.model_path = model_path
        # model
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path)
            print(f"已经加载{self.model_path}处模型")
        else:
            self.model = RenjuModel()
            print("初始化模型")
        self.init_weight()  # 初始化模型参数
        if torch.cuda.is_available():
            # 把模型放置 device
            self.model = self.model.to(device=torch.device("cuda"))

        # loss
        self.loss_pi = torch.nn.CrossEntropyLoss()  # 用于策略损失
        self.loss_q = torch.nn.MSELoss(reduction='mean')  # 用于价值损失

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.03, weight_decay=0.001)  # 优化器Adam可动态调整学习率

    def init_weight(self):
        """
        Initialize network parameters
        对于 nn.Conv2d 模块（卷积层），使用 Kaiming 初始化方法，其中权重使用正态分布，偏置初始化为0。
        对于 nn.BatchNorm2d 模块（批归一化层），将权重初始化为1，偏置初始化为0。
        对于 nn.Linear 模块（全连接层），使用正态分布初始化权重，偏置初始化为0.01。
        """
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train(self, start_epoch, epochs, early_stopping_patience=2):
        renju_dataset = RenjuDataset()

        train_size = int(len(renju_dataset) * 0.8)
        validate_size = int(len(renju_dataset) * 0.2)
        test_size = len(renju_dataset) - validate_size - train_size
        train_dataset, validate_dataset, test_dataset = torch.utils.data.random_split(renju_dataset,
                                                                                      [train_size, validate_size,
                                                                                       test_size])

        # input在dataloader内部 放置到cuda上
        train_dataloader = dataloader.DataLoader(train_dataset,
                                                 batch_size=64,
                                                 shuffle=True,
                                                 num_workers=0)  # 需注意 在windows中num_workers只能是0，在linux中才能多进程

        validate_dataloader = dataloader.DataLoader(validate_dataset,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    num_workers=0)  # 需注意 在windows中num_workers只能是0，在linux中才能多进程
        print("dataloader加载成功")
        best_validation_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            for i, item in enumerate(train_dataloader):
                s, p, q = item

                # forward
                predict_p, predict_q = self.model(s)
                # print(p.shape)
                # print(predict_p.view(-1))
                loss_pi = self.loss_pi(predict_p, p)

                loss_q = self.loss_q(predict_q.view(-1), q)  # 使用view将张量展平  即 [2,1] -> 2

                # backward
                self.optimizer.zero_grad()
                total_loss = loss_pi + loss_q
                total_loss.backward()
                self.optimizer.step()

                # Print or log the loss during training
                if (i + 1) % 500 == 0:
                    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], '
                          f'Loss_pi: {loss_pi.item():.4f}, Loss_q: {loss_q.item():.4f},total_loss:{total_loss:6.4f}\n"')

                # 保存loss
                with open("log" + os.sep + "loss.txt", "a") as file_point:
                    file_point.write(f"pi_loss:{loss_pi:6.3f}, q_loss:{loss_q:6.3f}, total_loss:{total_loss:6.4f}\n")
            print(f"Epoch {epoch + 1}/{epochs} completed.")

            # 在验证集上计算损失
            validation_loss = self.calculate_validation_loss(validation_dataloader=validate_dataloader)

            # 保存最好的模型
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(self.model, f"model" + os.sep + f"best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {validation_loss:.4f}")

            # 判断是否早停
            if patience_counter >= early_stopping_patience:
                print("Early stopping: No improvement in validation loss for {} epochs.".format(patience_counter))
                break

            # 保存模型
            torch.save(self.model, f"model" + os.sep + f"train_model{start_epoch + epoch}.pt")

    def calculate_validation_loss(self, validation_dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for i, item in enumerate(validation_dataloader):  # 使用验证集的dataloader
                s, p, q = item
                predict_p, predict_q = self.model(s)
                loss_pi = self.loss_pi(predict_p, p)
                loss_q = self.loss_q(predict_q.view(-1), q)
                total_loss += (loss_pi + loss_q).item()
        return total_loss / len(validation_dataloader)

    def fix_bn(self,m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def predict(self, x):
        self.model.eval()
        self.model.apply(self.fix_bn)
        with torch.no_grad():
            p, q = self.model(x)
        return p, q


def onedim_to_alphabet(moves: list):
    """
    将形如   [112, 113, 111, 110, 128, 96, 82, 127]
    转换形如: ['h8', 'h9', 'h7', 'h6', 'i9', 'g7', 'f8', 'i8]
    """
    results = []
    for move in moves:
        x, y = chr(ord('a') + move // 15), str((move % 15) + 1)
        res = x + y
        results.append(res)

    return results


if __name__ == "__main__":
    train = TrainNet("./model/train_model10_end.pt")
    train.train(0, epochs=20)  # 训练20轮
