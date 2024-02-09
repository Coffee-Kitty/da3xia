import torch
from torch import nn
from torchvision import models

"""
如果你没有显式地初始化神经网络模型的权重，
PyTorch会使用默认的初始化方法
线性层通常使用“Xavier”或“Glorot”初始化，
卷积层则使用“Kaiming”或“He”初始化。
"""


class myCNN(nn.Module):

    def get_name(self):
        return "cnn"

    def __init__(self, dropout_rate=0.3):
        super(myCNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class myResNet18(nn.Module):
    def get_name(self):
        return "resnet18"

    def __init__(self, num_classes=11):
        super(myResNet18, self).__init__()
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 移除原模型的最后一层全连接层
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

        # 替换为适合新分类任务的全连接层
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class myResNet50(nn.Module):
    def get_name(self):
        return "resnet50"

    def __init__(self, num_classes=11):
        super(myResNet50, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 移除原模型的最后一层全连接层
        self.features = nn.Sequential(*list(resnet50.children())[:-1])

        # 替换为适合新分类任务的全连接层
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class myResNet101(nn.Module):
    def get_name(self):
        return "resnet101"

    def __init__(self, num_classes=11):
        super(myResNet101, self).__init__()
        resnet101 = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # 移除原模型的最后一层全连接层
        self.features = nn.Sequential(*list(resnet101.children())[:-1])

        # 替换为适合新分类任务的全连接层
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



if __name__ == '__main__':
    # 创建模型实例
    custom_cnn = myCNN()
    custom_resnet = myResNet101()
    #
    # 打印模型的输入和输出维度
    input_tensor = torch.randn((2, 3, 128, 128))  # 1个样本，3个通道，128x128的图像
    output = custom_resnet(input_tensor)
    #
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    #
    # output = custom_cnn(input_tensor)
    #
    # print(f"Input shape: {input_tensor.shape}")
    # print(f"Output shape: {output.shape}")
    # print(custom_cnn.get_name())
    # print(custom_resnet.get_name())
    print(custom_resnet)
