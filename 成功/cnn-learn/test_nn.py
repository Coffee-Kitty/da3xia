"""
Test
"""
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configure import Configuration
from model import myCNN


def test_produce_csv(configure: Configuration):
    test_set = configure.test_set
    test_loader = configure.test_loader
    device = configure.device
    exp_name = configure.exp_name

    model_best = configure.model
    model_best.load_state_dict(torch.load(f"{exp_name}_best.ckpt"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    # create test csv
    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)

    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(len(test_set))]
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
