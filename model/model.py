import torch.nn as nn
import torch.nn.functional as F 


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = out.view(-1, 64*7*7)
        out = self.sigmoid(self.fc1(out))
        out = self.fc2(out)
        
        return out