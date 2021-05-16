import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from utils import train_model


class MLP(nn.Module):

    def __init__(self,layer_size=16):
        """
        Goal:
        Inputs:
        layer_size = int - number of hidden units per hidden layers
        Outputs:
        """
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
        """
        Goal:
        Compute the forward pass 
        Inputs:
        input - torch tensor - size Nx2 (N number of datapoints)
        Outputs:
        """
        output = self.sequence(input)
        return output

def train_pytorch_model(model,train_inputs,train_targets,
                        epochs=100,batch_size=50,lr=1e-1):
    """
    Goal:
    Train pytorch model
    Inputs:
    model = nn.Module object - model to be trained
    train_inputs = torch tensor - Nx2 (N number of data points)
    train_targets = torch.tensor - Nx1 (N number of data points)
    epochs = int - number of epochs for the training
    batch_size = int - batch size for the SGD
    lr = float - learning rate for the SGD
    Outputs:
    """                    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for b in range(0,train_inputs.shape[0],batch_size):
            inputs = train_inputs.narrow(0, b, batch_size)
            output = model(inputs)
            target = train_targets.narrow(0, b, batch_size)
            loss = criterion(output,target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()

def generate_images(train_set,train_target,
                    model,model_torch,steps,
                    epochs,folder="figures"):
    """
    Goal:
    Compare the the framework with pytorch using the contour prediction.
    Inputs:
    input - torch tensor - size Nx2 (N number of datapoints)
    Outputs:
    """
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

def generate_contours(model,epochs=100,save_file=None):
    """
    Goal:
    Given model from the framework, it will plot the decision boundaries/ prediction of the 
    model using a contour plot. 
    Inputs:
    model = framework object - 
    epochs = int -
    save_file = string - name file for the graph
    Outputs:
    """
    X = torch.linspace(0,1,1000)
    Y = torch.linspace(0,1,1000)
    grid_x, grid_y = torch.meshgrid(X,Y) # Grid of the input space
    grid_x_vector = grid_x.reshape(-1,1) # Flatten the grid
    grid_y_vector = grid_y.reshape(-1,1)
    inputs = torch.cat((grid_x_vector,grid_y_vector),dim=1)
    predicted = model.forward(inputs,no_grad=True)
    predicted = predicted.reshape(grid_x.shape[0],-1) # Reshape as a grid
    # Plot the graph 
    fig = plt.figure(figsize=[9,7])
    ax1 = fig.add_subplot(111)
    cs1 = ax1.contourf(grid_x,grid_y,predicted,cmap="viridis")
    plt.colorbar(cs1,ax=ax1)
    ax1.set_title("Framework predictions (epochs = " + str(epochs) + ")")
    if save_file is not None:
        fig.savefig("figures/" + save_file + ".svg")