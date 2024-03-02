import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models


class Model(nn.Module):
    """
    添加外部输入：

    基本思路：
    将原模型添加输入位置前的部分作为一个整体，同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系

    实现要点：
    torchvision中的resnet50输出是一个1000维的tensor，我们通过修改forward函数（配套定义一些层），
    先将1000维的tensor通过激活函数层和dropout层，再和外部输入变量"add_variable"拼接，
    最后通过全连接层映射到指定的输出维度10
    """

    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)), 1)
        x = self.fc_add(x)
        x = self.output(x)
        return x


net = models.resnet50()

net = Model(net).cuda()

# 修改输出：
# 假设我们要用这个resnet模型去做一个10分类的问题，就应该修改模型的fc层，将其输出节点数替换为10。
# 另外，我们觉得一层全连接层可能太少了，想再加一层。
classifier = nn.Sequential(
    OrderedDict(
        [
            ("fc1", nn.Linear(2048, 128)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(0.5)),
            ("fc2", nn.Linear(128, 10)),
            ("output", nn.Softmax(dim=1)),
        ]
    )
)

net.fc = classifier

print(net)
