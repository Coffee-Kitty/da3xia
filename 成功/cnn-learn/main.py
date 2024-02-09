import os

import torch.nn

from configure import Configuration
from model import myCNN, myResNet18, myResNet50, myResNet101
from test_nn import test_produce_csv
from train_nn import train
from visual_nn import vis_nn

if __name__ == "__main__":
    myCNN_config = Configuration(model=myCNN())

    train(myCNN_config)

    test_produce_csv(myCNN_config)

    vis_nn(myCNN_config)

    # myResNet18_config = Configuration(model=myResNet18())

    # train(myResNet18_config)

    # test_produce_csv(myResNet18_config)

    # myResNet50_config = Configuration(model=myResNet50())

    # train(myResNet50_config)

    # test_produce_csv(myResNet50_config)

    # myResNet101_config = Configuration(model=myResNet101())
    #
    # train(myResNet101_config)

    # test_produce_csv(myResNet101_config)
