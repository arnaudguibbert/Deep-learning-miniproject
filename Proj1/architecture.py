import torch
import torch.nn as nn

<<<<<<< HEAD
def create_naive_net():
    model = nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Flatten(),
        nn.Linear(128, 2)
    )
    return model
=======
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

    def forward(self,input):
        output = self.sequence(input)
        return output
>>>>>>> 9d98d90f25ae7e18348867cce8d9f8cbba7422e8
