from Linear import Linear
from MSELoss import MSELoss
from ReLU import ReLU
from Sequential import Sequential
from Tanh import Tanh
from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torch import empty
import torch
from pytorchNet import train_pytorch_model

def generate_disc_set(nb=1000):
    train_set = empty(nb, 2).uniform_()
    train_target = train_set - empty(1).fill_(0.5)
    train_target = (train_target.pow(2).sum(dim=1) < 1/(2*pi))*1
    train_target[train_target == 0] = -1
    return train_set, train_target.view(-1,1)

def create_model(nb_layers=3, layer_size=16):
    fc1 = Linear(2, layer_size)
    tanh = Tanh()
    relu = ReLU()
    layers_list = []
    for _ in range(nb_layers):
        fc = Linear(layer_size, layer_size)
        layers_list.append(fc)
        layers_list.append(relu)
    fc2 = Linear(layer_size, 1)
    sequence = [fc1, relu] + layers_list + [fc2, tanh]
    return Sequential(sequence)

def compute_nb_errors(model, data_input, data_target, pytorch=False):
    if pytorch:
        with torch.no_grad():
            output = model(data_input)
    else:
        output = model.forward(data_input, no_grad=True)
    output[output < 0] = -1
    output[output >= 0] = 1
    nb_errors = ((data_target != output)*1).sum()
    accuracy = (1 - nb_errors/data_input.shape[0])*100
    return accuracy.item()

def train_model(model,train_inputs,train_targets,
                epochs=100,mini_batch_size=100,lr=5e-2):
    criterion = MSELoss()
    for _ in range(epochs):
        for b in range(0,train_inputs.shape[0],mini_batch_size):
            inputs = train_inputs.narrow(0, b, mini_batch_size)
            output = model.forward(inputs)
            target = train_targets.narrow(0, b, mini_batch_size)
            loss = criterion.forward(output,target)
            grdwrtoutput= criterion.backward()
            model.zero_grad()
            model.backward(grdwrtoutput)
            model.optimization_step(lr)

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