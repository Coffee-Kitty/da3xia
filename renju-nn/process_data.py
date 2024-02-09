import xml.etree.ElementTree as ET
import json


def alphabet_to_onedim(moves: list):
    """
    将形如   ['h8', 'h9', 'h7', 'h6', 'i9', 'g7', 'f8', 'i8]
    转换形如: [112, 113, 111, 110, 128, 96, 82, 127]
    """
    results = []
    for move in moves:
        x, y = move[0], move[1]
        res = (ord(x) - ord('a')) * 15 + int(y) - 1
        results.append(res)

    return results


def save_file(games_data):
    # 将数据保存到 JSON 文件
    output_filename = './data/games_data.json'
    with open(output_filename, 'w') as json_file:
        json.dump(games_data, json_file)

    print(f"提取的信息已保存到 '{output_filename}' 中。")
    print(f"共计{len(games_data)}条数据")


# def load_file():
#     input_filename = './data/games_data.json'
#     with open(input_filename, 'r') as json_file:
#         games_data = json.load(json_file)
#         print(f"信息提取成功")
#         return games_data


def parse_data():
    # 解析XML文件
    tree = ET.parse('./renjunet_v10_20240128.rif')
    root = tree.getroot()

    # 获取所有游戏
    games = root.findall(".//games/game")

    games_data = []
    for game in games:
        # 获取游戏的一些属性，例如ID和结果
        game_id = game.get('id')
        result = game.get('bresult')

        if game.find("./move").text is None:
            continue
        # 获取游戏中的移动信息
        moves = alphabet_to_onedim(game.find("./move").text.split())
        if len(moves) == len(set(moves)):
            # 将信息添加到列表
            game_data = {
                "id": game_id,
                "result": result,
                "moves": moves
            }
            games_data.append(game_data)

    save_file(games_data)


parse_data()
# print(load_file())