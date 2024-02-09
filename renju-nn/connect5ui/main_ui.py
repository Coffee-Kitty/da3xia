import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QPixmap, QPainter, QPaintEvent, QPen, QMouseEvent, QFont
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QMessageBox, QPushButton, QInputDialog

sys.path.append(os.path.dirname(__file__))
from ai import MyAI
from board import Board, BOARD_LEN, WHITE, BLACK, EMPTY, NO_CLEAR_WINNER
from three_swap_hash import get_three_swap_or_not
from ui.board_widget import BOARD_UI_LEN, GRID_LEN, LEFT_MARGIN, TOP_MARGIN
from ui.design import Ui_MainWindow

# 时间常数
DEADLINE_TIME = 2147480000


def load_board(his: list) -> Board:
    """
    将history[] 元素为0-224加载到board
    #todo 对history中动作进行校验
    """
    board = Board()
    for tmp in his:
        if board.board[tmp] != EMPTY:
            print(f"有一个子落子重复,该子为{tmp}")
        else:
            board.play(tmp)
    return board


jsonFile = 'config.json'
with open(jsonFile, 'r', encoding='utf-8') as myfile:
    jsonText = myfile.read()
mixer = json.loads(jsonText)


class MainUi(QWidget):
    """
    长星局：[(7, 7), (7, 8), (6, 7)]
    峡月局：[(7, 7), (8, 8), (8, 6)]
    恒星局：[(7, 7), (6, 8), (8, 8)]
    水月局：[(7, 7), (7, 6), (6, 8)]
    流星局：[(7, 7), (8, 6), (6, 8)]
    云月局：[(7, 7), (6, 6), (8, 8)]
    浦月局：[(7, 7), (6, 8), (8, 6)]
    岚月局：[(7, 7), (6, 7), (8, 7)]
    银月局：[(7, 7), (7, 8), (8, 6)]
    明星局：[(7, 7), (6, 8), (7, 8)]
    斜月局：[(7, 7), (8, 8), (7, 6)]
    名月局：[(7, 7), (7, 6), (8, 8)]
    彗星局：[(7, 7), (8, 6), (6, 7)]
    寒星局：[(7, 7), (7, 6), (7, 8)]
    溪月局：[(7, 7), (7, 8), (6, 6)]
    疏星局：[(7, 7), (6, 6), (8, 8)]
    花月局：[(7, 7), (8, 8), (6, 6)]
    残月局：[(7, 7), (6, 6), (7, 8)]
    雨月局：[(7, 7), (7, 8), (8, 7)]
    金星局：[(7, 7), (8, 7), (6, 7)]
    松月局：[(7, 7), (6, 7), (8, 7)]
    丘月局：[(7, 7), (8, 6), (7, 6)]
    新月局：[(7, 7), (7, 6), (6, 7)]
    瑞星局：[(7, 7), (6, 7), (7, 8)]
    山月局：[(7, 7), (8, 7), (7, 6)]
    游星局：[(7, 7), (7, 8), (8, 8)]
    注意按以上顺序组织
    """

    def __init__(self, board: Board, ai: MyAI = None):
        super().__init__()

        # ai
        self.ai = ai
        # self.is_ai_first = mixer["is_ai_first"]
        self.is_ai_first = True
        # 首先查看引擎  如果有ai引擎 先等待ai引擎加载完毕
        if self.ai is not None:
            self.ai.demo()

        self.main_window = QMainWindow()
        self.main_window.setParent(self)

        self.design_ui = Ui_MainWindow()
        self.design_ui.setupUi(self.main_window)

        self.design_ui.left_widget.board = board

        self.design_ui.action_first.triggered.connect(self.ai_action_first)
        self.design_ui.action_second.triggered.connect(self.ai_action_second)

        # 棋盘辅助显示标签
        self.number_label = []
        self.alphabet_label = []
        for i in range(15):
            self.number_label.append(QLabel(f"{i + 1}", self.main_window))
            self.number_label[i].setGeometry(LEFT_MARGIN, TOP_MARGIN + 20 + int((BOARD_LEN - 1 - i) * GRID_LEN),
                                             GRID_LEN // 2, GRID_LEN // 2)
            self.number_label[i].setStyleSheet("background:transparent")
            tmp = chr(i + 65)
            self.alphabet_label.append(QLabel(tmp, self.main_window))
            self.alphabet_label[i].setGeometry(LEFT_MARGIN + int(i * GRID_LEN), int(BOARD_LEN * GRID_LEN),
                                               GRID_LEN // 2, GRID_LEN // 2)
            self.alphabet_label[i].setStyleSheet("background:transparent")

        # 计时器

        self.timer = QTimer(self)
        # self.timer.timeout.connect(self.time_has_timeout)  # 超时绑定的槽函数
        self.timer.singleShot(10, self.timer_show)  # 达到指定时间触发一次的槽函数
        # self.timer.start(DEADLINE_TIME)  # 先设置10秒
        # 显示时间label
        self.time_label = QLabel(self.main_window)
        self.time_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.time_label.setGeometry(BOARD_UI_LEN + 50, 50, 350, 25)
        self.time_label.setStyleSheet("background:white")

        self.start_time = time.time()  # 记录开始时间
        self.elapsed_time = 0

        # 敌方计时
        self.enemy_timer_label = QLabel(self.main_window)
        self.enemy_timer_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.enemy_timer_label.setStyleSheet("background:white")
        self.enemy_timer_label.setGeometry(BOARD_UI_LEN + 50, 125, 350, 25)

        self.enemy_elapsed_time = 0

        # # 总计时器
        # self.total_timer = QTimer(self)
        # self.total_timer.singleShot(10, self.total_time_show)  # 达到指定时间触发一次的槽函数
        # self.total_time_label = QLabel(self.main_window)
        # self.total_time_label.setStyleSheet("background:white")
        # self.total_time_label.setGeometry(BOARD_UI_LEN + 50, 125, 350, 25)
        # self.total_time_label.setFont(QFont("Roman times", 14, QFont.Bold))
        # self.total_start_time = time.time()  # 记录开始时间

        # 显示双方
        self.enemy_label = QLabel(self.main_window)
        self.my_ai_label = QLabel(self.main_window)
        self.enemy_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.my_ai_label.setFont(QFont("Roman times", 14, QFont.Bold))
        if self.is_ai_first:
            self.enemy_label.setText(f"{mixer['opposite_name']}： 白棋  后手")
            self.my_ai_label.setText(f"{mixer['my_name']}： 黑棋  先手")
        else:
            self.enemy_label.setText(f"{mixer['opposite_name']}： 黑棋  先手")
            self.my_ai_label.setText(f"{mixer['my_name']}： 白棋  后手")
        self.my_ai_label.setGeometry(BOARD_UI_LEN + 50, 250, 350, 25)
        self.enemy_label.setGeometry(BOARD_UI_LEN + 50, 200, 350, 25)
        self.my_ai_label.setStyleSheet("background:white")
        self.enemy_label.setStyleSheet("background:white")

        # 显示对局情况
        self.black_label = QLabel(self.main_window)
        self.white_label = QLabel(self.main_window)
        self.black_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.white_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.black_label.setStyleSheet("background:white")
        self.white_label.setStyleSheet("background:white")
        self.black_label.setGeometry(BOARD_UI_LEN + 50, 350, 350, 25)
        self.white_label.setGeometry(BOARD_UI_LEN + 50, 300, 350, 25)

        # 五手n打
        self.five_nums = 2  # 默认2打

        self.five_n_result = []

        # 三手交换
        self.un_play_button = QPushButton(self.main_window)
        self.un_play_button.setFont(QFont("Roman times", 14, QFont.Bold))
        self.un_play_button.setStyleSheet("background:white")
        self.un_play_button.setText("三手交换")
        self.un_play_button.setGeometry(BOARD_UI_LEN + 50, 450, 350, 40)
        self.un_play_button.clicked.connect(self.three_swap)
        self.three_swap_flag = False

        # 悔棋按钮
        self.un_play_button = QPushButton(self.main_window)
        self.un_play_button.setFont(QFont("Roman times", 14, QFont.Bold))
        self.un_play_button.setStyleSheet("background:white")
        self.un_play_button.setText("悔棋")
        self.un_play_button.setGeometry(BOARD_UI_LEN + 50, 550, 350, 50)
        self.un_play_button.clicked.connect(self.un_play)

        # 指定26种开局
        for i in range(26):
            self.design_ui.action[i].triggered.connect(lambda checked, idx=i: self.designate_position(idx))

        # 是否已经结束
        self.isAllOver = False

        # nps和winrate显示
        self.nps_label = QLabel(self.main_window)
        self.win_rate_label = QLabel(self.main_window)
        self.nps_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.win_rate_label.setFont(QFont("Roman times", 14, QFont.Bold))
        self.nps_label.setGeometry(BOARD_UI_LEN + 50, 650, 350, 25)
        self.win_rate_label.setGeometry(BOARD_UI_LEN + 50, 700, 350, 25)

    def mousePressEvent(self, a0: QMouseEvent) -> None:
        # 屏蔽ai线程运行时的鼠标点击事件
        if self.ai is not None and self.ai.isRunning():
            print("ai正在运行")
            return
        # 存在ai时才加上三手交换和五手n打
        if self.ai is None:
            self.play(a0)
        else:
            # 三手交换
            if self.design_ui.left_widget.board.turn == 3:
                # 如果我方ai为先手，点击三手交换按钮即可
                # 这里处理的是如果我方ai为后手，调用 三手交换ai 判断是否进行三手交换
                if not self.is_ai_first and not self.three_swap_flag:  # 并且ai不是由三手交换来的  只能进行一次三手交换！！！
                    if not self.ai_three_swap():
                        self.play(a0)  # 如果ai判断无需进行三手交换那就落子就好了
                else:
                    self.play(a0)
            # 五手n打
            elif self.design_ui.left_widget.board.turn == 4:
                self.five_n_zhuo()
            else:
                self.play(a0)

    # def input_x(self):
    #     while True:
    #         x = int(input())
    #         self.play(x)

    def play(self, a0):
        """
        如果是奇数回合，并且ai先手则ai下棋
        如果是偶数回合，并且是ai后手则ai下棋
        否则点击位置下棋
        """
        # 有ai
        if self.ai is not None:
            # 将下棋情况显示在label history元素(第几回合,那方,落子位置0-255)
            if (self.design_ui.left_widget.board.turn % 2 == 0 and self.is_ai_first) \
                    or (self.design_ui.left_widget.board.turn % 2 == 1 and not self.is_ai_first):
                acts = [act for turn, who, act in self.design_ui.left_widget.board.history]  # (第几回合,那方,落子位置0-255)
                self.ai.set_arg_to_return_one(acts)
                self.ai.result_ready.connect(self.handle_result_oneplay)
                self.ai.start()
            else:
                self.design_ui.left_widget.mousePressEvent(a0)
                if len(self.design_ui.left_widget.board.history) == 0:
                    return
                turn, who, act = self.design_ui.left_widget.board.history[-1]
                x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
                if who == BLACK:
                    self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
                elif who == WHITE:
                    self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")
                # 下完棋查看己方是否已经胜利,如果是黑方，查看是否违规失败，在main_ui调用查看
                win_lose_continue = self.design_ui.left_widget.board.game_over_who_win()
                if win_lose_continue == BLACK:
                    self.has_game_over(lose=WHITE)
                    return
                elif win_lose_continue == WHITE:
                    self.has_game_over(lose=BLACK)
                    return
                elif win_lose_continue == NO_CLEAR_WINNER:
                    self.has_game_over(lose=NO_CLEAR_WINNER)
                    return
        else:  # 没有ai
            self.design_ui.left_widget.mousePressEvent(a0)
            if len(self.design_ui.left_widget.board.history) == 0:
                return
            turn, who, act = self.design_ui.left_widget.board.history[-1]
            x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
            if who == BLACK:
                self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
            elif who == WHITE:
                self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")
            # 下完棋查看己方是否已经胜利,如果是黑方，查看是否违规失败，在main_ui调用查看
            win_lose_continue = self.design_ui.left_widget.board.game_over_who_win()
            if win_lose_continue == BLACK:
                self.has_game_over(lose=WHITE)
                return
            elif win_lose_continue == WHITE:
                self.has_game_over(lose=BLACK)
                return
            elif win_lose_continue == NO_CLEAR_WINNER:
                self.has_game_over(lose=NO_CLEAR_WINNER)
                return

    def handle_result_oneplay(self, result: list, nps: str, win_rate: str):
        # 如已经有棋子，ai输掉
        # if self.design_ui.left_widget.board.board[result] != EMPTY:
        #     if self.is_ai_first:
        #         self.has_game_over(lose=BLACK)
        #     else:
        #         self.has_game_over(lose=WHITE)
        #     return'
        self.nps_label.setText(nps)
        self.win_rate_label.setText(win_rate)
        result = result.pop()
        self.design_ui.left_widget.play(result)
        # 将下棋情况显示在label history元素(第几回合,那方,落子位置0-255)
        turn, who, act = self.design_ui.left_widget.board.history[-1]
        x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
        if who == BLACK:
            self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
        elif who == WHITE:
            self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")
        # 下完棋查看己方是否已经胜利,如果是黑方，查看是否违规失败，在main_ui调用查看
        win_lose_continue = self.design_ui.left_widget.board.game_over_who_win()
        if win_lose_continue == BLACK:
            self.has_game_over(lose=WHITE)
            return
        elif win_lose_continue == WHITE:
            self.has_game_over(lose=BLACK)
            return
        elif win_lose_continue == NO_CLEAR_WINNER:
            self.has_game_over(lose=NO_CLEAR_WINNER)
            return

    def handle_result_nplay(self, result: list, nps: str, win_rate: str):
        self.nps_label.setText(nps)
        self.win_rate_label.setText(win_rate)
        end_x, end_y = 0, 0
        action_set = result
        resstr = "{"
        for tmp_one_x in action_set:
            x = tmp_one_x // BOARD_LEN
            y = tmp_one_x % BOARD_LEN
            resstr += "(" + chr(x + 65) + "," + str(y + 1) + ")"
        resstr += "}"
        mes = QMessageBox()
        mes.setText(f"五手N打根据N打数量{self.five_nums}{mixer['my_name']}黑棋ai选择落点为:" + resstr)
        mes.exec_()

        flag = True
        act_str = ""
        for tmp_one_x in action_set:
            x = tmp_one_x // BOARD_LEN
            y = tmp_one_x % BOARD_LEN
            act_str += "(" + chr(65 + x) + "," + str(y + 1) + ")"
        while flag:
            text, ok = QInputDialog.getText(self, '五手N打',
                                            f"请{mixer['opposite_name']}白方选择{self.five_nums}打{act_str}的一个落点:(例: (A,1)):")
            if ok:
                text = str(text)
                try:
                    x, y = text.strip('()').split(',')
                    x = x.strip().upper()  # 将x坐标转换为大写字母
                    y = int(y.strip()) - 1
                    if 'A' <= x <= 'Z':
                        x = ord(x) - ord('A')  # 将字母A-Z转换为0-25的数字
                    else:
                        raise ValueError("输入必须为A-Z")
                except ValueError as e:
                    # 处理输入格式错误的异常
                    print("输入格式错误:", e)
                    continue
                if x * BOARD_LEN + y not in action_set:
                    flag = True
                else:
                    flag = False
                    end_x, end_y = x, y

        one_x = end_x * BOARD_LEN + end_y
        self.design_ui.left_widget.board.play(one_x)
        self.ai.disconnect()

        # 将下棋情况显示在label history元素(第几回合,那方,落子位置0-255)
        turn, who, act = self.design_ui.left_widget.board.history[-1]
        x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
        if who == BLACK:
            self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
        elif who == WHITE:
            self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")
        # 下完棋查看己方是否已经胜利,如果是黑方，查看是否违规失败，在main_ui调用查看
        win_lose_continue = self.design_ui.left_widget.board.game_over_who_win()
        if win_lose_continue == BLACK:
            self.has_game_over(lose=WHITE)
            return
        elif win_lose_continue == WHITE:
            self.has_game_over(lose=BLACK)
            return
        elif win_lose_continue == NO_CLEAR_WINNER:
            self.has_game_over(lose=NO_CLEAR_WINNER)
            return

    def timer_show(self):
        if self.design_ui.left_widget.board.turn % 2 == 0:
            all_elapse_time = time.time() - self.start_time  # 记录结束时间
            self.elapsed_time = all_elapse_time - self.enemy_elapsed_time
            minutes = self.elapsed_time // 60
            seconds = self.elapsed_time % 60
            self.time_label.setText(f"黑棋 总计时{minutes:.2f}分{seconds:.2f}秒")
        elif self.design_ui.left_widget.board.turn % 2 == 1:
            all_elapse_time = time.time() - self.start_time  # 记录结束时间
            self.enemy_elapsed_time = all_elapse_time - self.elapsed_time
            minutes = self.enemy_elapsed_time // 60
            seconds = self.enemy_elapsed_time % 60
            self.enemy_timer_label.setText(f"白棋 总计时{minutes:.2f}分{seconds:.2f}秒")
        # 重新触发即可
        self.timer.singleShot(10, self.timer_show)

    # def total_time_show(self):
    #     elapsed = time.time() - self.total_start_time
    #     minutes = elapsed // 60
    #     seconds = elapsed % 60
    #     self.total_time_label.setText(f"总计时:{minutes:.2f}分{seconds:.2f}秒")
    #     self.total_timer.singleShot(10, self.total_time_show)

    # def time_has_timeout(self):
    #     """
    #     到时间后返回对方胜利即可
    #     """
    #     if self.design_ui.left_widget.board.player == WHITE:
    #         self.has_game_over(WHITE)
    #     else:
    #         self.has_game_over(BLACK)
    #     # 关闭定时器
    #     self.timer.stop()

    def has_game_over(self, lose):

        if self.isAllOver:
            return

        if not self.isAllOver:
            self.isAllOver = True
        mes = ''
        if lose == WHITE:
            mes = "黑棋胜，是否保存棋盘"
        elif lose == BLACK:
            mes = "白棋胜，是否保存棋盘"
        else:
            mes = "平局，是否保存棋盘"

        message = QMessageBox.question(self, "终局", mes, QMessageBox.Yes, QMessageBox.No)
        if message == QMessageBox.Yes:
            # 用户点击了"是"按钮
            # 在此执行相应的操作
            # 以下是打印棋谱，无用
            if not os.path.exists('./board_history'):
                os.makedirs('./board_history')
            current_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime())
            hex_data = ''
            file_name = ''
            my_name = mixer["my_name"]
            oppo_name = mixer["opposite_name"]
            if self.is_ai_first:
                if lose == WHITE:
                    winner = '先手胜'
                elif lose == BLACK:
                    winner = '后手胜'
                elif lose == NO_CLEAR_WINNER:
                    winner = '平局'
                hex_data += '{' + f'[C5][{my_name} B][' + oppo_name + ' W][' + winner + '][' + str(
                    current_time) + ' 线上][2023 CCGC]'
                file_name += f'./board_history/C5-{my_name} B vs ' + oppo_name + ' W-' + winner + '胜-网上' + f'{current_time}' + '-CCGC.txt'
            else:
                if lose == WHITE:
                    winner = '先手胜'
                elif lose == BLACK:
                    winner = '后手胜'
                elif lose == NO_CLEAR_WINNER:
                    winner = '平局'
                hex_data += '([C5][' + oppo_name + f' B][{my_name} W][' + winner + '][' + str(
                    current_time) + ' 线上][2023 CCGC]'
                file_name += './board_history/C5-' + oppo_name + f' B vs {my_name} W-' + winner + '胜-网上' + f'{current_time}' + '-CCGC.txt'
            hex_data = str(hex_data)
            for turn, player, act in self.design_ui.left_widget.board.history:
                if player == BLACK:
                    hex_data += ';B(' + str(chr(65 + (act // BOARD_LEN))) + ',' + str(
                        (act % BOARD_LEN)) + ')'
                else:
                    hex_data += ';W(' + str(chr(65 + (act // BOARD_LEN))) + ',' + str(
                        act % BOARD_LEN) + ')'
            hex_data += '}'
            # 获取当前工作目录
            current_directory = Path.cwd()
            # 文件名
            file_name = file_name.replace(":", "_").replace(" ", "_")
            # 文件路径
            file_path = current_directory / file_name

            with open(file_path, 'w', encoding='gb2312') as output:
                output.write(hex_data)
                output.write('\n')
                output.close()

        else:
            # 用户点击了"否"按钮或关闭了对话框
            # 在此执行相应的操作
            pass
        # self.close()

    def designate_position(self, i):
        """
        26种指定开局的槽函数
        """

        # 五手n打也需要ai指定...
        # 如果我方ai先手,先默认为2
        self.isAllOver = False
        self.three_swap_flag = False
        if self.is_ai_first:
            self.five_nums = 2
            mess = QMessageBox()
            mess.setText(f"{mixer['my_name']}ai黑棋选择五手{self.five_nums}打")
            mess.exec_()
        else:
            text, ok = QInputDialog.getText(self, '五手N打', '请输入第五手打点数量(例: 3):')
            if ok:
                self.five_nums = int(str(text))

        openings = [
            # 这里坐标为 y,x 请注意
            # 斜指开局

            # 长星局
            [(7, 7), (8, 8), (9, 9)],
            # 峡月局
            [(7, 7), (8, 8), (8, 9)],
            # 恒星局
            [(7, 7), (8, 8), (7, 9)],
            # 水月局
            [(7, 7), (8, 8), (6, 9)],
            # 流星局
            [(7, 7), (8, 8), (5, 9)],
            # 云月局
            [(7, 7), (8, 8), (7, 8)],
            # 浦月局
            [(7, 7), (8, 8), (6, 8)],
            # 岚月局
            [(7, 7), (8, 8), (5, 8)],
            # 银月局
            [(7, 7), (8, 8), (6, 7)],
            # 明星局
            [(7, 7), (8, 8), (5, 7)],
            # 斜月局
            [(7, 7), (8, 8), (6, 6)],
            # 名月局
            [(7, 7), (8, 8), (5, 6)],
            # 彗星局
            [(7, 7), (8, 8), (5, 5)],

            # 直指开局
            # 寒星局
            [(7, 7), (8, 7), (9, 7)],
            # 溪月局
            [(7, 7), (8, 7), (9, 8)],
            # 疏星局
            [(7, 7), (8, 7), (9, 9)],
            # 花月局
            [(7, 7), (8, 7), (8, 8)],
            # 残月局
            [(7, 7), (8, 7), (8, 9)],
            # 雨月局
            [(7, 7), (8, 7), (7, 8)],
            # 金星局
            [(7, 7), (8, 7), (7, 9)],
            # 松月局
            [(7, 7), (8, 7), (6, 7)],
            # 丘月局
            [(7, 7), (8, 7), (6, 8)],
            # 新月局
            [(7, 7), (8, 7), (6, 9)],
            # 瑞星局
            [(7, 7), (8, 7), (5, 7)],
            # 山月局
            [(7, 7), (8, 7), (5, 8)],
            # 游星局
            [(7, 7), (8, 7), (5, 9)],

        ]
        history = [(x * BOARD_LEN + y) for y, x in openings[i]]
        new_board = load_board(history)

        self.design_ui.left_widget.board = new_board
        turn, who, act = self.design_ui.left_widget.board.history[-1]
        x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
        if who == BLACK:
            self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
        elif who == WHITE:
            self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")

        self.start_time = time.time()
        self.elapsed_time = 0
        self.enemy_elapsed_time = 0
        self.time_label.setText(f"黑棋 总计时{0:.2f}分{0:.2f}秒")
        self.enemy_timer_label.setText(f"白棋 总计时{0:.2f}分{0:.2f}秒")
        self.repaint()

    def three_swap(self):
        """
        三手交换
        """
        if self.ai is not None and self.ai.isRunning():
            return

        if self.design_ui.left_widget.board.turn == 3:
            self.is_ai_first = not self.is_ai_first
            self.three_swap_flag = True

            if self.is_ai_first:
                self.enemy_label.setText(f"{mixer['opposite_name']}： 白棋  后手")
                self.my_ai_label.setText(f"{mixer['my_name']}： 黑棋  先手")
            else:
                self.enemy_label.setText(f"{mixer['opposite_name']}： 黑棋  先手")
                self.my_ai_label.setText(f"{mixer['my_name']}： 白棋  后手")

    def ai_three_swap(self) -> bool:
        # 如果是第3回合 进行ai三手交换的判断
        turn1, player1, act1 = self.design_ui.left_widget.board.history[0]
        turn2, player2, act2 = self.design_ui.left_widget.board.history[1]
        turn3, player3, act3 = self.design_ui.left_widget.board.history[2]
        flag = get_three_swap_or_not(act1, act2, act3, self.five_nums)
        if flag:
            self.three_swap()
            message = QMessageBox()
            message.setText(f"{mixer['my_name']}一方决定进行三手交换")
            message.exec_()
            return True
        else:
            # ai不进行三手交换那就下棋呗
            return False

    def un_play(self):

        # 如果ai在跑，不能悔棋
        if self.ai is not None and self.ai.isRunning():
            return

        self.design_ui.left_widget.un_play()
        if len(self.design_ui.left_widget.board.history) == 0:
            return
        turn, who, act = self.design_ui.left_widget.board.history[-1]
        x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
        if who == BLACK:
            self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
        elif who == WHITE:
            self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")

    def five_n_zhuo(self):
        """
        五手N打
        :return:
        """
        end_x, end_y = 0, 0
        # 如果我方ai是黑棋，由ai根据n打的数量，指定
        if self.is_ai_first:
            if self.design_ui.left_widget.board.turn == 4:
                # 如果我方是黑棋并且没有进行三手交换的话 我方默认疏星局开局 且指定打点(由神经网络得出)
                if not self.three_swap_flag:
                    result = [6 * 15 + 8, 9 * 15 + 8]
                    action_set = result
                    resstr = "{"
                    for tmp_one_x in action_set:
                        x = tmp_one_x // BOARD_LEN
                        y = tmp_one_x % BOARD_LEN
                        resstr += "(" + chr(x + 65) + "," + str(y + 1) + ")"
                    resstr += "}"
                    mes = QMessageBox()
                    mes.setText(f"五手N打根据N打数量{self.five_nums}{mixer['my_name']}黑棋ai选择落点为:" + resstr)
                    mes.exec_()

                    flag = True
                    act_str = ""
                    for tmp_one_x in action_set:
                        x = tmp_one_x // BOARD_LEN
                        y = tmp_one_x % BOARD_LEN
                        act_str += "(" + chr(65 + x) + "," + str(y + 1) + ")"
                    while flag:
                        text, ok = QInputDialog.getText(self, '五手N打',
                                                        f"请{mixer['opposite_name']}白方选择{self.five_nums}打{act_str}的一个落点:(例: (A,1)):")
                        if ok:
                            text = str(text)
                            try:
                                x, y = text.strip('()').split(',')
                                x = x.strip().upper()  # 将x坐标转换为大写字母
                                y = int(y.strip()) - 1
                                if 'A' <= x <= 'Z':
                                    x = ord(x) - ord('A')  # 将字母A-Z转换为0-25的数字
                                else:
                                    raise ValueError("输入必须为A-Z")
                            except ValueError as e:
                                # 处理输入格式错误的异常
                                print("输入格式错误:", e)
                                continue
                            if x * BOARD_LEN + y not in action_set:
                                flag = True
                            else:
                                flag = False
                                end_x, end_y = x, y

                    one_x = end_x * BOARD_LEN + end_y
                    self.design_ui.left_widget.board.play(one_x)
                    # 将下棋情况显示在label history元素(第几回合,那方,落子位置0-255)
                    turn, who, act = self.design_ui.left_widget.board.history[-1]
                    x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
                    if who == BLACK:
                        self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
                    elif who == WHITE:
                        self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")
                    # 下完棋查看己方是否已经胜利,如果是黑方，查看是否违规失败，在main_ui调用查看
                    win_lose_continue = self.design_ui.left_widget.board.game_over_who_win()
                    if win_lose_continue == BLACK:
                        self.has_game_over(lose=WHITE)
                        return
                    elif win_lose_continue == WHITE:
                        self.has_game_over(lose=BLACK)
                        return
                    elif win_lose_continue == NO_CLEAR_WINNER:
                        self.has_game_over(lose=NO_CLEAR_WINNER)
                        return

                else:
                    acts = [act for turn, who, act in self.design_ui.left_widget.board.history]
                    self.ai.set_arg_to_return_n(self.five_nums, acts)
                    self.ai.result_ready.connect(self.handle_result_nplay)
                    self.ai.start()

        # 如果我方ai是白棋，需选取黑方指定的棋
        elif not self.is_ai_first:
            two_dim_actions = []
            if self.design_ui.left_widget.board.turn == 4:
                flag = True
                while flag:
                    text, ok = QInputDialog.getText(self, '五手N打',
                                                    f"根据N打数量{self.five_nums}请{mixer['opposite_name']}方黑棋选择落点为:(格式: (A,1),(B,1)):")
                    if ok:
                        coord_str = str(text)
                        coordinates = coord_str.strip('()').split('),(')

                        for coordinate in coordinates:
                            try:
                                x, y = coordinate.split(',')
                                x = x.strip().upper()  # 将x坐标转换为大写字母
                                y = int(y.strip()) - 1
                                if 'A' <= x <= 'Z':
                                    x = ord(x) - ord('A')  # 将字母A-Z转换为0-25的数字
                                else:
                                    raise ValueError("x坐标必须为A到Z之间的大写字母")
                                if y < 0 or y >= 15:
                                    raise ValueError("x坐标必须为A到Z之间的大写字母")
                                two_dim_actions.append((x, y))
                            except ValueError as e:
                                # 处理输入格式错误的异常
                                print("输入格式错误:", e)
                                two_dim_actions.clear()  # 清空动作集合，准备重新收集输入
                                break  # 跳出内部循环，重新开始外部循环
                        if len(two_dim_actions) != self.five_nums:
                            flag = True
                            two_dim_actions.clear()
                        else:
                            flag = False

                act_str = ""
                for act in two_dim_actions:
                    act_str += "(" + chr(65 + act[0]) + "," + str(act[1] + 1) + ")"

                # end_x, end_y = two_dim_actions.pop()
                actions = [x * 15 + y for x, y in two_dim_actions]
                pre_actions = [pre_act for pre_turn, pre_player, pre_act in self.design_ui.left_widget.board.history]

                self.ai.five_n_zhuo_best_select.connect(self.handle_five_n_zhuo)

                self.ai.get_lowest_rank(act_str, pre_actions, actions)
                self.ai.start()

    def handle_five_n_zhuo(self, act_str, end_x, end_y):
        """
        异步处理来自ai的 五手n打选的最差的点  act_str为对方给点坐标
        :param end_x:
        :param end_y:
        :return:
        """
        mes = QMessageBox()
        mes.setText(f"五手N打{mixer['my_name']}白方选择N打{act_str}的一个落点:({chr(end_x + 65)},{end_y + 1})")
        mes.exec_()

        one_x = end_x * BOARD_LEN + end_y
        self.design_ui.left_widget.board.play(one_x)
        # 将下棋情况显示在label history元素(第几回合,那方,落子位置0-255)
        turn, who, act = self.design_ui.left_widget.board.history[-1]
        x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
        if who == BLACK:
            self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
        elif who == WHITE:
            self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")
        # 下完棋查看己方是否已经胜利,如果是黑方，查看是否违规失败，在main_ui调用查看
        win_lose_continue = self.design_ui.left_widget.board.game_over_who_win()
        if win_lose_continue == BLACK:
            self.has_game_over(lose=WHITE)
            return
        elif win_lose_continue == WHITE:
            self.has_game_over(lose=BLACK)
            return
        elif win_lose_continue == NO_CLEAR_WINNER:
            self.has_game_over(lose=NO_CLEAR_WINNER)
            return

    def ai_action_first(self):
        self.is_ai_first = True
        self.enemy_label.setText(f"{mixer['opposite_name']}： 白棋  后手")
        self.my_ai_label.setText(f"{mixer['my_name']}： 黑棋  先手")

        # 黑棋我方以疏星局开局  打点打两个
        self.five_nums = 2
        openings = [
            # 疏星局
            [(7, 7), (8, 7), (9, 9)],
        ]
        history = [(x * BOARD_LEN + y) for y, x in openings[0]]
        new_board = load_board(history)

        self.design_ui.left_widget.board = new_board
        turn, who, act = self.design_ui.left_widget.board.history[-1]
        x, y = chr(65 + act // BOARD_LEN), act % BOARD_LEN
        if who == BLACK:
            self.black_label.setText(f"黑:第{turn + 1}回合落到{(x, y + 1)}")
        elif who == WHITE:
            self.white_label.setText(f"白:第{turn + 1}回合落到{(x, y + 1)}")

        self.start_time = time.time()
        self.elapsed_time = 0
        self.enemy_elapsed_time = 0
        self.time_label.setText(f"黑棋 总计时{0:.2f}分{0:.2f}秒")
        self.enemy_timer_label.setText(f"白棋 总计时{0:.2f}分{0:.2f}秒")
        self.repaint()

    def ai_action_second(self):
        self.is_ai_first = False
        self.enemy_label.setText(f"{mixer['opposite_name']}： 黑棋  先手")
        self.my_ai_label.setText(f"{mixer['my_name']}： 白棋  后手")
