import torch
import torch.nn as nn
import torch.nn.functional as F


class Naive_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
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


class MnistCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,kernel_size=(3,3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,128,kernel_size=(3,3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,10)
        )

    def forward(self,input):
        output = self.sequence(input)
        return output


class CrossArchitecture(nn.Module):

    def __init__(self):
        super().__init__()
        self.MnistNet = MnistCNN()
        self.NaiveNet = Naive_net()
        self.target_type = ["target0","target1"]
        self.Linear1 = nn.Linear(22,11)
        self.Linear2 = nn.Linear(11,2)
        self.weights_loss = [0.5,0.5]
    
    def forward(self,input):
        num1 = input[:,[0],:,:]
        num2 = input[:,[1],:,:]
        output_naive = self.NaiveNet(input)
        output_num1 = self.MnistNet(num1).view(num1.shape[0],-1,1)
        output_num2 = self.MnistNet(num2).view(num2.shape[0],-1,1)
        output2 = torch.cat((output_num1,output_num2),dim=2)
        output_num1 = output_num1.view(num1.shape[0],-1)
        output_num2 = output_num2.view(num2.shape[0],-1)
        output1 = torch.cat((output_num1,output_num2,output_naive),dim=1)
        output1 = self.Linear1(output1)
        output1 = self.Linear2(output1)
        return output1, output2