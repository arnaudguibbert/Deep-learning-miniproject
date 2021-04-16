import torch
import torch.nn as nn

class Naive_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        self.target_type = ["target0"]
        self.weights_loss = [1]

    def forward(self,input):
        output = self.sequence(input)
        return output


class Naive_net_2(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        self.target_type = ["target0"]
        self.weights_loss = [1]

    def forward(self,input):
        output = self.sequence(input)
        return output
