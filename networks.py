import torch
import torch.nn.functional as F
from torch import nn

# Convolutional Neural Network
class CNN(nn.Module):
    def __init__(self, inference=False):
        super(CNN, self).__init__()
        self.inference = inference
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        if self.inference:
            x = x.reshape(280, 280, 4)
            x = torch.narrow(x, dim=2, start=3, length=1)
            x = x.reshape(1, 1, 280, 280)
            x = F.avg_pool2d(x, 10, stride=10)
            x = x / 255
            x = (x - mean_val) / standard_deviation_val
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        if self.inference:
            return F.softmax(x, dim=1)
        else:
            return x