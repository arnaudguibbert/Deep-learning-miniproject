import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from utils import train_model


class MLP(nn.Module):

    def __init__(self,layer_size=16):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(2, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,1),
            nn.Tanh()
        )

    def forward(self,input):
        output = self.sequence(input)
        return output

def train_pytorch_model(model,train_inputs,train_targets,
                        epochs=100,mini_batch_size=100,lr=5e-2):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for b in range(0,train_inputs.shape[0],mini_batch_size):
            inputs = train_inputs.narrow(0, b, mini_batch_size)
            output = model(inputs)
            target = train_targets.narrow(0, b, mini_batch_size)
            loss = criterion(output,target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()

def generate_images(train_set,train_target,
                    model,model_torch,steps,
                    epochs,folder="figures"):
    X = torch.linspace(0,1,1000)
    Y = torch.linspace(0,1,1000)
    grid_x, grid_y = torch.meshgrid(X,Y)
    grid_x_vector = grid_x.reshape(-1,1)
    grid_y_vector = grid_y.reshape(-1,1)
    inputs = torch.cat((grid_x_vector,grid_y_vector),dim=1)
    for nb_epochs in range(steps,epochs+1,steps):
        train_model(model,train_set,train_target,epochs=steps)
        train_pytorch_model(model_torch,train_set,train_target,epochs=steps)
        predicted = model.forward(inputs,no_grad=True)
        predicted = predicted.reshape(grid_x.shape[0],-1)
        with torch.no_grad():
            predicted_torch = model_torch(inputs)
            predicted_torch = predicted_torch.reshape(grid_x.shape[0],-1)
        fig = plt.figure(figsize=[16,7])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.contourf(grid_x,grid_y,predicted)
        ax2.contourf(grid_x,grid_y,predicted_torch)
        ax1.set_title("Our framework")
        ax2.set_title("Pytorch")
        fig.suptitle(str(nb_epochs) + " epochs")
        fig.savefig(folder + "/epochs" + str(nb_epochs) + ".jpg", dpi=250)