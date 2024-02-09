# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QAction

from ui.board_widget import BoardWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1200, 800)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setAutoFillBackground(False)
#         MainWindow.setStyleSheet("\n"
# "background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,stop: 0 #FF626E, stop: 1.0 #FFBE71);\n"
# "")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.left_widget = BoardWidget(self.centralwidget)
        self.left_widget.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.left_widget.setStyleSheet("\n"
"background: transparent;")
        self.left_widget.setObjectName("left_widget")
        self.horizontalLayout.addWidget(self.left_widget)
        self.right_widget = QtWidgets.QWidget(self.centralwidget)
        self.right_widget.setEnabled(False)
        self.right_widget.setStyleSheet("\n"
"background:transparent;")
        self.right_widget.setObjectName("right_widget")
        self.horizontalLayout.addWidget(self.right_widget)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # 假设这是26种对局的名字列表
        opening_names = [
            # 斜指开局
            '长星局',
            '峡月局',
            '恒星局',
            '水月局',
            '流星局',
            '云月局',
            '浦月局',
            '岚月局',
            '银月局',
            '明星局',
            '斜月局',
            '名月局',
            '彗星局',

            # 直指开局
            '寒星局',
            '溪月局',
            '疏星局',
            '花月局',
            '残月局',
            '雨月局',
            '金星局',
            '松月局',
            '丘月局',
            '新月局',
            '瑞星局',
            '山月局',
            '游星局'
        ]

        self.action = []
        for i, name in enumerate(opening_names):
            # 创建QAction并设置字体样式
            action = QAction(name, self.menu)
            self.action.append(action)
            font = QFont("Arial", 12)  # 替换成您想要的字体和大小
            action.setFont(font)
            # 将QAction添加到菜单
            self.menu.addAction(action)
        self.menubar.addAction(self.menu.menuAction())


        # ai先后手的指定
        self.is_ai_first_menu = QtWidgets.QMenu("指定先后手", self.menubar)
        self.action_first = QAction("ai先手,我方为黑棋")
        self.action_first.setFont(QFont("Arial", 12))
        self.action_second = QAction("ai后手,我方为白棋")
        self.action_second.setFont(QFont("Arial", 12))

        self.is_ai_first_menu.addAction(self.action_first)
        self.is_ai_first_menu.addAction(self.action_second)
        self.menubar.addAction(self.is_ai_first_menu.menuAction())


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "悟空五子棋"))
        self.menu.setTitle(_translate("MainWindow", "开始"))



