import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

from dataset import FoodDataset
from model import myCNN, myResNet50


class Configuration:
    def __init__(self, model=myResNet50()):
        myseed = 6666  # set a random seed for reproducibility
        torch.backends.cudnn.deterministic = True  # 启用了 cuDNN 的确定性模式，这将禁用一些可能引入的非确定性操作，有助于实验的可重复性
        torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的 benchmark 模式，这通常会在训练开始时优化卷积操作，但会导致结果的非确定性
        np.random.seed(myseed)
        torch.manual_seed(myseed)
        if torch.cuda.is_available():
            print("cuda is is_available")
            torch.cuda.manual_seed_all(myseed)
        """
        Configurations
        """
        # "cuda" only when GPUs are available.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Initialize a model, and put it on the device specified.

        # 检测gpu数目
        print(f"of  cuda counts in this configure is :  {torch.cuda.device_count()}  ")

        self.model = model.to(self.device)
        if os.path.exists(f"./{self.model.get_name()}_best.ckpt"):
            self.model.load_state_dict(torch.load(f"{self.model.get_name()}_best.ckpt"))
            print(f"load model:   {self.model.get_name()}_best.ckpt")
        else:
            print(f" init model {self.model.get_name()} ")
        self.exp_name = self.model.get_name()

        # The number of batch size.
        self.batch_size = 32
        # The number of training epochs.
        self.n_epochs = 24
        # If no improvement in 'patience' epochs, early stop.
        self.patience = 3
        # For the classification task, we use cross-entropy as the measurement of performance.
        self.criterion = nn.CrossEntropyLoss()
        # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)

        # 验证集准确率
        self.best_acc = 0
        if os.path.exists(f"./{self.model.get_name()}_best_acc.txt"):
            with open(f"./{self.model.get_name()}_best_acc.txt", "r") as f:
                self.best_acc = float(f.readline())
                print(f"load best_acc : {self.best_acc}")


        """
            数据预处理
        """
        # 测试集无需数据增强，只需把图片转换为tensor
        test_tfm = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        # 对训练集进行数据增强
        train_tfm = transforms.Compose([
            # Resize the image into a fixed shape (height = width = 128)
            transforms.Resize((128, 128)),

            # 自动增强
            transforms.RandomChoice(transforms=[
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.Lambda(lambda x: x)
            ], p=[0.98, 0.02]),

            # ToTensor() should be the last one of the transforms.
            transforms.ToTensor(),
        ])
        """
        Dataloader
        """
        # Construct train and valid datasets.
        # The argument "loader" tells how torchvision reads the data.
        train_set = FoodDataset("./food-11/train", tfm=train_tfm)
        # num_workers 指定数据加载的并行工作数量，设置为 0 表示不使用多进程加载数据  pin_memory=True 表示将数据加载到 CUDA 异步内存中，可以加速 GPU 训练
        self.train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)
        valid_set = FoodDataset("./food-11/valid", tfm=test_tfm)
        self.valid_loader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                       pin_memory=True)
        self.test_set = FoodDataset("./food-11/test", tfm=test_tfm)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True)

if __name__ == "__main__":
    with open(f"./resnet50_best_acc.txt", "w") as f:
        f.write(f"{0.222}")
        f.close()