import numpy as np
import torch

from tensor_board import history_to_feature
from train import TrainNet, onedim_to_alphabet

train = TrainNet("./model/best_model.pt")
his = [112,113,110,128,111,143,109,158]
feature = history_to_feature(his)
p, q = train.predict(feature.to(torch.device("cuda")))

p = p.view(-1).detach().cpu().numpy()
q = q.detach().cpu().numpy()
print(p.sum())
# print(p)
# 获取前 10 个最高值的索引
# top_10_indices = np.argsort(-p)[:10].tolist()
top_10_indices = np.argsort(-p).tolist()
top_10_act = onedim_to_alphabet(top_10_indices)
print(his)
print(top_10_act)
# 根据索引获取前 10 个最高值
top_10_values = p[top_10_indices]
print(top_10_values)
print(q)
