import torch
import torch.nn as nn

class Minst_cnn(nn.Module):
    def __init__(self,num_classes=10):
        super(Minst_cnn,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = out.view(-1, 64*7*7)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
__all__ = ['Minst_cnn']