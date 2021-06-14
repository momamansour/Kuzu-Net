# kuzu.py
# CSE, UNSW
# Author: Mohammed Almansour

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        # INSERT CODE HERE
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.log_softmax(self.fc1(x))
        return x # CHANGE CODE HERE
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.tanh(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x # CHANGE CODE HERE
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.fc1 = nn.Linear(32 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))   # -1 conforms to the other dimension
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x # CHANGE CODE HERE
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
