import torch.nn as nn


# Squeeze and Excitation Networks的核心SE Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        # 全局平均池化，输入B*C*H*W -> 输出 B*C*1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 两个全连接层，输入B*C*1*1 -> B*(C/r)*1*1 -> B*C*1*1，带来更多的非线性，更好地拟合通道相关性
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()

        # B*C*1*1转成B*C，再送入FC层
        y = self.avg_pool(x).view(bs, c)

        # 两个全连接层+激活
        y = self.fc(y).view(bs, c, 1, 1)

        # 和原特征图相乘
        return x * y.expand_as(x)
    

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from utlis import summary
    se = SEBlock(channel=512,reduction=8)
    summary(se, (512, 8, 8), device="cpu")