import torch
import torch.nn as nn
import torch.nn.functional as F

class BigNaive(nn.Module):
    """
    Input: Nx2x14x14 images
    Output: Nx2 labels
    
    CNN with the following architecture (BN and Dropout not mentioned):
    2x14x14 -> (conv) 32x12x12 -> (conv) 64x10x10 -> (conv) 64x8x8
    -> (maxpool) 64x4x4 -> (conv) 64x2x2 -> (maxpool) 64x1x1
    -> (flatten) 64 -> (fc) 16 -> (fc) 2
    """
    
    def __init__(self):
        super().__init__()
        
        self.sequence = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(1/6),
            
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(1/4),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(),
            
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(64),
            nn.Dropout(),
            nn.Flatten(),
            
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Linear(16, 2)
        )
        
        self.target_type = ["target0"]
        self.weights_loss = [1]
        
    def forward(self,input):
        output = self.sequence(input)
        return output
            


class Naive_net(nn.Module):
    """
    Input: Nx2x14x14 images
    Output: Nx2 labels
    
    CNN with the following architecture (BN and Dropout not mentioned):
    2x14x14 -> (conv) 48x12x12 -> (maxpool) 48x6x6 -> (conv) 32x4x4 
    -> (maxpool) 32x2x2 -> (flatten) 128 -> (fc) 64 -> (fc) 2
    """

    def __init__(self,artifice=0):
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
    """
    Input: Nx1x14x14
    Output: Nx10
    
    CNN for digit classification of one channel with the following architecture:
    1x14x14 -> (conv) 96x12x12 -> (maxpool) 96x6x6 -> (conv) 48x5x5
    -> (conv) 32x4x4 -> (maxpool) 32x2x2 -> (conv) 64x1x1 -> (flatten) 64
    -> (fc) 128 -> (fc) 64 -> (fc) 10
    """

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self.sequence = nn.Sequential(
            nn.Conv2d(1,96,kernel_size=(3,3)),
            self.activation,
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(96),

            nn.Dropout(p=0.65),
            
            nn.Conv2d(96,48,kernel_size=(2,2)),
            self.activation,
            nn.BatchNorm2d(48),

            nn.Dropout(p=0.25),

            nn.Conv2d(48,32,kernel_size=(2,2)),
            self.activation,
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)),
            nn.BatchNorm2d(32),
        
            nn.Dropout(p=0.25),

            nn.Conv2d(32,64,kernel_size=(2,2)),
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
    """
    Creates a resblock with out dim = in dim
    """
    
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
    """
    Input: Nx1x14x14
    Output: Nx10
    
    ResNet for digit classification with architecture (BN, Dropout not mentioned):
    1x14x14 -> (conv) 32x11x11 -> (conv) 16x8x8
    -> (resblocks) 16x8x8 -> (maxpool) 16x4x4 -> (conv) 16x2x2
    -> (flatten) 64 -> (fc) 10
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
        return output
        

class Simple_Net(nn.Module):
    """
    Another CNN for digit classification, similar architecture as MnistCNN
    Inspired by: https://github.com/Coderx7/SimpleNet
    """

    def __init__(self):
        super().__init__()
        self.sequence = nn.Sequential(
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
    """
    First attempt to combine a CNN for digit-classification and a CNN for comparison (binary output).
    Two losses, one for each task. Concatenates an intermediate output of the digit-classification 
    part to the comparison part, hence the name CrossArchitecture.
    
    Input: Nx2x14x14
    Output: (Nx10), (Nx2)
    """

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
        if self.training:
            return output1, output2
        else:
            return output1

class oO_Net(nn.Module):
    """
    Input: Nx2x14x14
    Output: (Nx10), (Nx2)
    
    Combines two arms with two loss functions: one arm for digit-classification, one for binary classif.
    binary classification is done by using "Naive_net"
    digit-classification can be done either by a CNN or a ResNet (set use_MnistResNet to True)
    Information is shared between both arms by concatenation or summation
    """
    
    def __init__(self, Nb_ResBlocks=1,embedded_dim=2,use_MnistResNet=False,weights_loss=[0.2,0.8]):
        super().__init__()
        self.target_type = ["target0","target1"]
        self.weights_loss = weights_loss
        self.emb_dim=embedded_dim
        self.Nb_ResBlocks=Nb_ResBlocks

        if use_MnistResNet:
            nb_blocks = self.Nb_ResBlocks
            self.Mnist_part = MnistResNet().sequence[:13+nb_blocks] # out shape = (N,64,2)
        else:
            self.Mnist_part = MnistCNN().sequence[:24] # out shape = (N,64,2)
            
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
            nn.Linear(32,self.emb_dim)
        )
        
        self.lower_last_sequence = nn.Sequential(
            nn.Linear(20+self.emb_dim,12),
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
        
        if self.training:
            return output_down, output_up

        else:
            return output_down


class ResNextBlock(nn.Module):

    def __init__(self,n_parrallel=4):
        super().__init__()
        self.blocks = ["block" + str(i) for i in range(n_parrallel)]
        for block in self.blocks:
            sequence = nn.Sequential(
                nn.Conv2d(2,2,kernel_size=(3,3),padding=(1,1)),
                nn.BatchNorm2d(2),
                nn.ReLU()
            )
            setattr(self,block,sequence)

    def forward(self,input):
        output = input
        for block in self.blocks:
            output = output+getattr(self,block)(input)
        return output


class LugiaNet(nn.Module):
    """
    Input: Nx2x14x14
    Output: (Nx10), (Nx2)
    """
    def __init__(self,n_parrallel=2):
        super().__init__()
        self.target_type = ["target0","target1","target1"]
        self.weights_loss = [0.3,0.4,0.3]
        self.resblock = ResNextBlock(n_parrallel=n_parrallel)
        self.Mnist = MnistCNN()
        self.FC_naive = nn.Sequential(nn.Linear(36,8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8,2))
        self.Naive = Naive_net().sequence[:11]
        self.post_naive_sequence = nn.Sequential(
            nn.Linear(128,16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16,8),
        )

    def forward(self,input):
        nb_sample = input.shape[0]
        height, width = input.shape[2], input.shape[3]
        output_resblock = self.resblock(input)
        mnist_input_resblock = output_resblock.reshape(-1,1,
                                                       height,
                                                       width)
        mnist_input_original = input.reshape(-1,1,
                                             output_resblock.shape[2],
                                             output_resblock.shape[3])
        mnist_output_resblock = self.Mnist(mnist_input_resblock).reshape(nb_sample,2,10).permute(0,2,1)
        mnist_output_original = self.Mnist(mnist_input_original).reshape(nb_sample,2,10).permute(0,2,1)
        output_naive_original = self.post_naive_sequence(self.Naive(input))
        output_naive_resblock = self.post_naive_sequence(self.Naive(output_resblock))
        input_FC_naive = torch.cat((output_naive_original,
                                    output_naive_resblock,
                                    mnist_output_original.reshape(-1,20)),dim=1)
        output_FC_naive = self.FC_naive(input_FC_naive)

        if self.training:
            return output_FC_naive, mnist_output_resblock, mnist_output_original
        else:
            return output_FC_naive





