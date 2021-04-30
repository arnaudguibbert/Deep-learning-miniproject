import torch
import torch.nn as nn
import torch.nn.functional as F


class Naive_net(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(),

            nn.Conv2d(48, 32, 3),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.Dropout(),
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
        self.activation = nn.ReLU()
        self.sequence = nn.Sequential(
            nn.Conv2d(1,96,kernel_size=(3,3)),
            self.activation,
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(96),

            nn.Dropout(p=0.65),
            
            nn.Conv2d(96,48,kernel_size=(3,3)),
            self.activation,
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(48),
        
            nn.Dropout(p=0.25),

            nn.Conv2d(48,64,kernel_size=(2,2)),
            self.activation,
            nn.BatchNorm2d(64),
            
            nn.Dropout(p=0.25),

            nn.Flatten(),
            nn.Linear(64,128),
            self.activation,
            nn.BatchNorm1d(128),
            
            nn.Linear(128,64),
            self.activation,
            nn.BatchNorm1d(64),
            nn.Linear(64,10)
        )

    def forward(self,input):
        output = self.sequence(input)
        return output
    
class ResBlock(nn.Module):
    
    def __init__(self,nb_channels,kernel_size):
        super().__init__()     
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
        padding = (kernel_size-1)//2)
        self.bn1 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size,
        padding = (kernel_size-1)//2)
        self.bn2 = nn.BatchNorm2d(nb_channels)
        
    def forward(self, x):
        y = self.bn1(self.conv1(x))
        y = F.relu(y)
        y = self.bn2(self.conv2(y))
        y += x
        y = F.relu(y)
        return y
        
class MnistResNet(nn.Module):
    """ResNet for MNIST with the following architecture:
    1x14x14 -> (conv) 32x11x11 -> (conv) 16x8x8
    -> (resblocks) 16x8x8 -> (maxpool) 16x4x4 -> (conv) 16x2x2
    -> (flatten) 64 -> (fc) 10
    Some dropout can/should be added rather at the end of the net
    """
    
    def __init__(self, nb_blocks=3):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.sequence = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size = 4),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            *(ResBlock(nb_channels=16, kernel_size=3) for _ in range(self.nb_blocks)),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,16,3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Flatten(),
            nn.Linear(64, 10)
        )
    
    def forward(self, input):
        output = self.sequence(input)
        return ouput
        

class Simple_Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
            #simpleNet inspired implementation :https://github.com/Coderx7/SimpleNet
            #Conv1
            nn.Conv2d(1,64,kernel_size=(3,3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(48,48,kernel_size=(3,3)), 
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.ReLU(),
            
            nn.Conv2d(48,64,kernel_size=(3,3)), 
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,kernel_size=(1,1)), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,64,kernel_size=(3,3)), 
            
            nn.Flatten(),
            nn.Linear(64,64),
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
        self.SimpleNet = Simple_Net()
        #target 1 is the class and target 0 is the comparison (??)
        self.target_type = ["target0","target1"]
        self.Linear1 = nn.Linear(22,11)
        self.Linear2 = nn.Linear(11,2)
        self.weights_loss = [0.5,0.5]
    
    def forward(self,input):
        """MnistCNN outputs a prediction for both digits, NaiveNet outputs the predicted comparison.
        No weights shared up to that point. Then, two linear layers take these predictions as input (size N*22).
        Output: predicted comparison of the two digits."""
        num1 = input[:,[0],:,:]
        num2 = input[:,[1],:,:]
        output_naive = self.NaiveNet(input) # shape = (N,2)
        output_num1 = self.MnistNet(num1).view(num1.shape[0],-1,1) # shape = (N,10,1)
        output_num2 = self.MnistNet(num2).view(num2.shape[0],-1,1) # shape = (N,10,1)
        output2 = torch.cat((output_num1,output_num2),dim=2) # shape = (N,10,2)
        output_num1 = output_num1.view(num1.shape[0],-1) # shape = (N,10)
        output_num2 = output_num2.view(num2.shape[0],-1) # shape = (N,10)
        output1 = torch.cat((output_num1,output_num2,output_naive),dim=1) # shape = (N,22)
        output1 = self.Linear1(output1)
        output1 = self.Linear2(output1) # shape = (N,2)
        return output1, output2

class oO_Net(nn.Module):
    
    def __init__(self, use_MnistResNet=False):
        super().__init__()
        self.target_type = ["target0","target1"]
        self.weights_loss = [0.5,0.5]
        
        if use_MnistResNet:
            nb_blocks = MnistResNet().nb_blocks
            self.Mnist_part = MnistResNet().sequence[:13+nb_blocks] # out shape = (N,64,2)
        else:
            self.Mnist_part = MnistCNN().sequence[:21] # out shape = (N,64,2)
            
        self.Naive_part = Naive_net().sequence[:11] # out shape = (N,128)

        self.post_mnist_sequence = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,10)
        )
        
        self.post_naive_sequence = nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32,4)
        )
        
        self.lower_last_sequence = nn.Sequential(
            nn.Linear(24,12),
            nn.ReLU(),
            nn.BatchNorm1d(12),
            nn.Linear(12,2)
        )
        
    def forward(self,input):
        num1 = input[:,[0],:,:]
        num2 = input[:,[1],:,:]
        upper_output1 = self.Mnist_part(num1).view(num1.shape[0],-1,1)
        upper_output2 = self.Mnist_part(num2).view(num2.shape[0],-1,1)
        upper_output = torch.cat((upper_output1,upper_output2),dim=2) # shape = (N,64,2)
        
        lower_output = self.Naive_part(input) # shape = (N,128)
        
        # Sum both outputs after making them shape-compatible
        upper_output = upper_output.view(upper_output.shape[0], -1) # shape = (N,128)
        lower_part, upper_part = lower_output + upper_output, upper_output + lower_output
        
        # Then, take upper and lower parts to sizes 10 and 2 respectively
        upper_part = upper_part.view(upper_part.shape[0],-1,2)
        upper_part1, upper_part2 = upper_part[:,:,0], upper_part[:,:,1] # restore the two channels
        
        upper_part1 = self.post_mnist_sequence(upper_part1)
        upper_part2 = self.post_mnist_sequence(upper_part2)
        
        lower_part = self.post_naive_sequence(lower_part)
        
        # Form the final outputs
        temp1, temp2 = upper_part1.view(upper_part1.shape[0], -1, 1), upper_part2.view(upper_part2.shape[0], -1, 1)
        output_up = torch.cat((temp1, temp2),dim=2)
        
        output_down = torch.cat((upper_part1,upper_part2,lower_part),dim=1)
        output_down = self.lower_last_sequence(output_down)
        
        return output_down, output_up