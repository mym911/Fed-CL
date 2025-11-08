from copy import deepcopy

import torch
from torch import nn


class Chead(torch.nn.Module):
    def __init__(self,label_num):
        super(Chead,self).__init__()

        # self.head = nn.Sequential(nn.Linear(768*2, label_num))
        self.head = nn.Sequential(nn.Linear(768*2, 512),
        # 激活函数
        nn.ReLU(),
        # Dropout 层，防止过拟合
        nn.Dropout(p=0.5),
        # 第二层全连接层
        nn.Linear(512, label_num))
    def forward(self,x):
        x = self.head(x)
        return x

    def load_head(self,head):
        self.head = deepcopy(head)

    def get_head(self):
        return self.head