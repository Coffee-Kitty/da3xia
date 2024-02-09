import sys

from PyQt5.QtWidgets import QApplication

from ai import MyAI
from main_ui import MainUi, load_board

if __name__ == "__main__":
    app = QApplication(sys.argv)

    history = [112, 113, 110]
    connect5_board = load_board(history)

    # ai = MyAI()

    connect5_ui = MainUi(connect5_board, ai=None)
    # connect5_ui = MainUi(connect5_board, ai=ai)
    connect5_ui.show()

    sys.exit(app.exec_())
