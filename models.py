import torch.nn as nn
import torch.nn.functional as F


class Network1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=12)
        self.fc1 = nn.Linear(in_features=(288), out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=2)

    def forward(self, t):
        t = t
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool1d(t, kernel_size=4)

        t = t.reshape(-1, 288)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.softmax(t)

        return t