from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen, QFont
from PyQt5.QtWidgets import QWidget

from board import BOARD_LEN, WHITE, BLACK, EMPTY, CONTINUE

LEFT_MARGIN = 30
TOP_MARGIN = 10
BOARD_UI_LEN = 750
GRID_LEN = int(BOARD_UI_LEN / BOARD_LEN)
# 线宽
LINE_WIDTH = 2


def one_dim_to_two_dim(one_x) -> (int, int):
    """
    将流式坐标 0-224 转换为 (x,y)坐标，且是ui上的(x,y)坐标
    """
    y, x = (one_x % BOARD_LEN, one_x // BOARD_LEN)
    y = BOARD_LEN - 1 - y
    return int(LEFT_MARGIN + x * GRID_LEN), int(TOP_MARGIN + y * GRID_LEN)


def two_dim_to_one_dim(x, y) -> int:
    """
    将ui文件的(x,y)坐标转换为 线性坐标0-225
    """
    x, y = (x - LEFT_MARGIN) // GRID_LEN, (y - TOP_MARGIN) // GRID_LEN
    y = BOARD_LEN - 1 - y
    one_x = int(x * BOARD_LEN + y)
    return one_x


def draw_circle(center_x: int, center_y: int, radius: int, painter: QPainter, color: int):
    """
    在(x,y)用color画一个半径为radius的实心圆
    :return:
    """
    painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿效果
    # 设置矩形的填充颜色为
    painter.setBrush(color)
    # 绘制实心圆形
    radius = radius  # 圆形半径
    x = center_x  # 圆心横坐标
    y = center_y  # 圆心纵坐标
    painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
    if color == Qt.white:
        painter.setPen(Qt.black)
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)


# def draw_chess(one_x, color, painter: QPainter):
#     """
#     画棋子(i,j)
#     """
#     x, y = one_dim_to_two_dim(one_x)
#     radius = 20
#     if color == WHITE:
#         color = Qt.white
#     elif color == BLACK:
#         color = Qt.black
#     painter.setPen(color)
#     draw_circle(x, y, radius, painter, color)


def draw_chess(one_x, color, index, painter):
    """
    画棋子(i,j)并显示阿拉伯数字
    """
    x, y = one_dim_to_two_dim(one_x)
    radius = 20

    if color == WHITE:
        color = Qt.white
    elif color == BLACK:
        color = Qt.black

    painter.setPen(color)
    draw_circle(x, y, radius, painter, color)

    # 设置字体
    font = QFont()
    font.setPointSize(12)
    painter.setFont(font)
    painter.setPen(Qt.red)
    # 在棋子中心绘制阿拉伯数字
    text_rect = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
    painter.drawText(text_rect, Qt.AlignCenter, str(index))

# 为最后一个落子加一个标志
def draw_last_chess_label(one_x, color, painter: QPainter):
    x, y = one_dim_to_two_dim(one_x)
    radius = 10
    # if color == WHITE:
    #     color = Qt.white
    # elif color == BLACK:
    #     color = Qt.black
    painter.setRenderHint(QPainter.Antialiasing)  # 抗锯齿效果
    painter.setPen(Qt.red)
    painter.setBrush(Qt.red)
    painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)


def draw_lines(painter: QPainter):
    """
    画15*15的棋盘
    """
    painter.setPen(QPen(Qt.black, LINE_WIDTH))
    v_nums = BOARD_LEN
    h_nums = BOARD_LEN
    for i in range(v_nums):
        start_x = int(LEFT_MARGIN + i * GRID_LEN)
        start_y = int(TOP_MARGIN)
        end_x = int(LEFT_MARGIN + i * GRID_LEN)
        end_y = int(TOP_MARGIN + BOARD_UI_LEN - GRID_LEN)
        painter.drawLine(start_x, start_y, end_x, end_y)
    for i in range(h_nums):
        start_x = int(LEFT_MARGIN)
        start_y = int(TOP_MARGIN + i * GRID_LEN)
        end_x = int(LEFT_MARGIN + BOARD_UI_LEN - GRID_LEN)
        end_y = int(TOP_MARGIN + i * GRID_LEN)
        painter.drawLine(start_x, start_y, end_x, end_y)


class BoardWidget(QWidget):

    def __init__(self, parent):
        super().__init__(parent)
        self.board = None
        self.changed = False  # 用于判断是否已经改变

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QPainter(self)
        draw_lines(painter)

        if self.board is not None:

            for pre_turn, pre_player, pre_act in self.board.history:
                draw_chess(pre_act,pre_player,pre_turn,painter)
            # for one_x, color in enumerate(self.board.board):
            #     if color != EMPTY:
            #         draw_chess(one_x, color, painter)


            if len(self.board.history) > 0:
                pre_turn, pre_player, pre_act = self.board.history[-1]
                draw_last_chess_label(pre_act, pre_player, painter)

        painter.end()

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        if a0.button() == Qt.LeftButton:
            x = int(a0.pos().x())
            y = int(a0.pos().y())
            # print(f"点击了位置：{(x, y)}")
            # 越界置之不理
            if x < LEFT_MARGIN or x > LEFT_MARGIN + BOARD_LEN * GRID_LEN - 10:
                return None
            if y < TOP_MARGIN or y > TOP_MARGIN + BOARD_LEN * GRID_LEN - 5:
                return None
            one_x = two_dim_to_one_dim(x, y)
            self.play(one_x)

    def un_play(self):
        if len(self.board.history) == 0:
            return
        self.board.un_play()
        self.changed = True
        self.repaint()

    def play(self, one_x):

        # 如已经有棋子，先置之不理 #todo
        if self.board.board[one_x] != EMPTY:
            return None

        self.board.play(one_x)
        # 下完棋查看己方是否已经胜利,如果是黑方，查看是否违规失败，在main_ui调用查看
        self.changed = True
        self.repaint()
