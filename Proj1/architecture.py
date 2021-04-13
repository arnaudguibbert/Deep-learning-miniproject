import torch
import torch.nn as nn

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