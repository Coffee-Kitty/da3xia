import os
import subprocess

import torch
from PyQt5.QtCore import QThread, pyqtSignal


# arguments = ['2', '2', '1' , '4']
# ret = subprocess.Popen(
#     r"./123/connect5-ai.exe",
#     stdout=subprocess.PIPE,
#     stdin=subprocess.PIPE,
#     stderr=subprocess.PIPE,
# )
#
# for tmp in arguments:
#     ret.stdin.write(bytes(tmp + "\n",'utf-8'))
#     ret.stdin.flush()
# result = []
# for i in range(2):
#     output = ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
#     result.append(int(output))
# print(result)
"""
# # 结果 41
# # """


# 如果我方是黑棋，并且五手n打 打点为(G,9)(J，9)
class MyAI(QThread):
    # 正常取最好的点
    result_ready = pyqtSignal(list, str, str)

    # 五手n打选取对方最差的点
    five_n_zhuo_best_select = pyqtSignal(str, int, int)

    def demo(self):
        """
        测试引擎是否已经装载完毕
        :return:
        """
        self.arguments = ['1', '1', '1', '1']
        for tmp in self.arguments:
            self.ret.stdin.write(bytes(tmp + "\n", 'utf-8'))
            self.ret.stdin.flush()
        result = []
        output = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
        print(output)
        output = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
        print(output)
        nps = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
        print(nps)
        winrate = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
        print(winrate)
        for i in range(self.n):
            output = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
            result.append(int(output))


        print("ai引擎加载完毕")

    def get_lowest_rank(self, designate_five_zhuos: str, board_arg: list, arg: list):
        """
        根据nn 得到在board_arg基础下 的arg种最坏的动作

        !!!这里是由纯神经网络选取的最差的dian ...


        :param board_arg:
        :param arg:
        :return:
        """
        self.arguments = ['200', str(len(board_arg))]
        for tmp in board_arg:
            str_tmp = str(tmp)
            self.arguments.append(str_tmp)
        self.arguments.append(str(len(arg)))
        for tmp in arg:
            str_tmp = str(tmp)
            self.arguments.append(str_tmp)
        for tmp in self.arguments:
            self.ret.stdin.write(bytes(tmp + "\n", 'utf-8'))
            self.ret.stdin.flush()

        output = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
        result = int(output)
        end_x, end_y = result // 15, result % 15
        self.five_n_zhuo_best_select.emit(designate_five_zhuos, end_x, end_y)
        return


    def __init__(self):
        super().__init__()
        self.path = r"Release/connect5-ai.exe"
        self.arguments = []
        self.n = 1
        self.ret = subprocess.Popen(
            self.path,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def set_arg_to_return_one(self, arg: list):
        """
        接受 [1, 2, 225] 类型参数 7换为str类型并设置
        """
        # self.arguments = ['50', '100', str(len(arg))]
        self.arguments = ['20', '1', str(len(arg))]
        self.n = 1
        for tmp in arg:
            str_tmp = str(tmp)
            self.arguments.append(str_tmp)

    def set_arg_to_return_n(self, n, arg: list):
        self.arguments = ['20', str(n), str(len(arg))]
        self.n = n
        for tmp in arg:
            str_tmp = str(tmp)
            self.arguments.append(str_tmp)


    def run(self):
        """
        返回预测的棋子位置 0-224之间
        """

        # 白棋疏星局开局 第四步最优点已经确定 不要再用网络搜索
        # 疏星局
        if len(self.arguments) == 6 and self.arguments[3] == '112' and self.arguments[4] == '113' and self.arguments[
            5] == '144':  # 112,113,144
            result = [128]
            self.result_ready.emit(result, '', '')
        else:
            for tmp in self.arguments:
                self.ret.stdin.write(bytes(tmp + "\n", 'utf-8'))
                self.ret.stdin.flush()
            result = []
            output = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
            print(output)
            output = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
            print(output)
            nps = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
            print(nps)
            winrate = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
            print(winrate)
            for i in range(self.n):
                output = self.ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
                result.append(int(output))

            self.result_ready.emit(result, nps, winrate)
        # print("ai给出的动作"+result)

    # 112,113,144,128,143

    def __del__(self):
        self.ret.kill()
        print("引擎子线程已经kill")
