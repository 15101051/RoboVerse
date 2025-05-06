import torch
import torch.nn as nn
import torch.nn.functional as F

class ExceptionDetector(nn.Module):
    def __init__(self):
        super(ExceptionDetector, self).__init__()
        self.fc1 = nn.Linear(1568, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 加入 Batch Normalization
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)  # 加入 Batch Normalization
        self.fc3 = nn.Linear(128, 1)

    def forward(self, global_cond):
        x = F.relu(self.bn1(self.fc1(global_cond)))  # 先进行归一化再激活
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
