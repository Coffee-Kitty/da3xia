# coding: utf-8
import glob
import json

datas = glob.glob("../../chinese-poetry-master/全唐诗/poet.song.*.json")

for data in datas:
    with open(data, 'r', encoding='utf-8') as fp:
        poems = json.load(fp)
        for poem in poems:
            if len(poem['paragraphs']) == 2 and \
                    len(poem['paragraphs'][0]) == 12 and \
                    len(poem['paragraphs'][1]) == 12:
                with open('tangshi.txt', 'a',encoding='utf-8') as f:
                    f.write("".join(poem['paragraphs']))
                    f.write("\n")

