import os
import subprocess
import time

from pathlib import Path

from board import Board, CONTINUE, BLACK, WHITE

connec_ret = subprocess.Popen(
    r"../Release/connect5-ai.exe",
    stdout=subprocess.PIPE,
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

renju_ret = subprocess.Popen(
    r"../Release-opposite/connect5_ai.exe",
    stdout=subprocess.PIPE,
    stdin=subprocess.PIPE,
    stderr=subprocess.PIPE,
)


def connect5AI(acts: list) -> int:
    arguments = ['15', '1', str(len(acts))]
    for tmp in acts:
        str_tmp = str(tmp)
        arguments.append(str_tmp)

    for tmp in arguments:
        connec_ret.stdin.write(bytes(tmp + "\n", 'utf-8'))
        connec_ret.stdin.flush()
    output = connec_ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    print(output)
    output = connec_ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    print(output)
    output = connec_ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    print(output)

    output = connec_ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    result = int(output)

    return result


def renjuAI(acts: list) -> int:
    arguments = ['15', str(len(acts))]
    for tmp in acts:
        str_tmp = str(tmp)
        arguments.append(str_tmp)

    for tmp in arguments:
        renju_ret.stdin.write(bytes(tmp + "\n", 'utf-8'))
        renju_ret.stdin.flush()

    output = renju_ret.stdout.readline().decode('utf-8')  # Read the output and decode from bytes to string
    result = int(output)

    return result


# renju为白方  connrct5 为黑方

play_count = 10
win_rate = 0.0
connect5_win = 0

while True:
    if play_count <= 0:
        break

    play_count -= 1
    board = Board()
    his = [112, 113, 144]
    board.play(112)
    board.play(113)
    board.play(144)

    while board.game_over_who_win() == CONTINUE:

        connect5_act = connect5AI(his)
        board.play(connect5_act)
        board.print_board()
        his.append(connect5_act)

        if board.game_over_who_win() != CONTINUE:
            break

        renju5_act = renjuAI(his)
        board.play(renju5_act)
        board.print_board()
        his.append(renju5_act)

    if board.game_over_who_win() == BLACK:
        print("黑方connect5胜利")
        connect5_win += 1
        win_rate = connect5_win / play_count
    if board.game_over_who_win() == WHITE:
        print("白方renju胜利")
        sstr = "{"
        for pre_turn, pre_player, pre_act in board.history:
            sstr += str(pre_act)
            if pre_act != board.history[-1][2]:
                sstr += ","
        sstr += "} "
        sstr += "nums:" + str(len(board.history))
        print(sstr)
        current_directory = Path.cwd()
        current_time = time.strftime('%Y.%m.%d %H:%M:%S', time.localtime())
        # 文件名
        if not os.path.exists('connect5_loss_as_black'):
            os.makedirs('connect5_loss_as_black')
        file_name = "./connect5_loss_as_black/" + current_time + ".txt"
        file_name = file_name.replace(":", "_").replace(" ", "_")
        # 文件路径
        file_path = current_directory / file_name

        with open(file_path, 'w', encoding='utf-8') as output:
            output.write(sstr)
            output.write('\n')
            output.close()

    print(f"最后一个落子位置为:{his[-1] // 15},{his[-1] % 15}")

    # board.print_board()
    print(f"胜利率{win_rate} 总场次{play_count}")

connec_ret.kill()
renju_ret.kill()
print(f"胜利率{win_rate} 总场次{play_count}")
