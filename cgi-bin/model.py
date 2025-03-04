"""
Define Convolutional Nerual Network model for MNIST input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.conv1 = nn.Conv2d(1,  32, kernel_size=5, padding="same")
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding="same"
)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding="same")
    self.fc1 = nn.Linear(6272, 256)
    self.fc2 = nn.Linear(256, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu(F.max_pool2d(self.conv3(x),2))
    x = F.dropout(x, p=0.5, training=self.training)
    x = torch.flatten(x, start_dim=1)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = F.relu(self.fc2(x))
    return F.log_softmax(x, dim=1)
