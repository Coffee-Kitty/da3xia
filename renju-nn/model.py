import torch
import torch.nn as nn
import torch.nn.functional as f


class ResidualBlock(nn.Module):
    """
    残差块
    """

    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),  # inplace=True 参数表示在原地进行操作
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel))
        self.right = nn.Sequential(
            # Use Conv2d with the kernel_size of 1, without padding to improve the parameters of the network
            nn.Conv2d(in_channel, out_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel))
        if torch.cuda.is_available():
            self.left.cuda()
            self.right.cuda()

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        return f.relu(out)


class RenjuModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.BOARD_SIZE = 15
        self.layer1 = ResidualBlock(3, 12)
        self.layer2 = ResidualBlock(12, 48)
        self.layer3 = ResidualBlock(48, 64)
        self.layer4 = ResidualBlock(64, 64)

        # policy network
        self.layer1_p = ResidualBlock(64, 64)
        self.layer2_p = ResidualBlock(64, 16)
        self.policy_fc = nn.Linear(16 * self.BOARD_SIZE * self.BOARD_SIZE, 512)
        self.policy_batch_norm = nn.LayerNorm(512)
        self.policy = nn.Linear(512, self.BOARD_SIZE * self.BOARD_SIZE)
        # value network
        self.layer1_v = ResidualBlock(64, 64)
        self.layer2_v = ResidualBlock(64, 16)
        self.value_fc = nn.Linear(16 * self.BOARD_SIZE * self.BOARD_SIZE, 64)
        self.value_batch_norm = nn.LayerNorm(64)
        self.value = nn.Linear(64, 1)

        if torch.cuda.is_available():
            self.policy_fc.cuda()
            self.policy_batch_norm.cuda()
            self.policy.cuda()
            self.value_fc.cuda()
            self.value_batch_norm.cuda()
            self.value.cuda()

    def forward(self, x):
        x = x.reshape(-1, 3, self.BOARD_SIZE, self.BOARD_SIZE)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # policy network
        pi = self.layer1_p(x)
        pi = self.layer2_p(pi)
        pi = pi.view(-1, 16 * self.BOARD_SIZE * self.BOARD_SIZE)
        pi = self.policy_fc(pi)
        pi = f.relu(self.policy_batch_norm(pi))
        pi = f.softmax(self.policy(pi), dim=1)
        # value network
        v = self.layer1_v(x)
        v = self.layer2_v(v)
        v = v.view(-1, 16 * self.BOARD_SIZE * self.BOARD_SIZE)
        v = self.value_fc(v)
        v = f.relu(self.value_batch_norm(v))
        v = torch.tanh(self.value(v))
        return pi, v


if __name__ == "__main__":
    input = torch.zeros((1, 3, 15, 15))
    if torch.cuda.is_available():
        input = input.to('cuda')
    model = RenjuModel()
    pi, v = model(input)
    print(pi.shape, v.shape)
    print(pi.view(-1).shape)
    print(v.view(-1).shape)
    # print(pi)
    # print(model)
