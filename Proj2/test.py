from Linear import *
from MSELoss import *
from RELu import *
from Sequential import *
from Tanh import *
import random
from torch import empty
from math import pi

def generate_disc_set(nb=1000):
    train_set = torch.empty(nb)._uniform()
    # the following is false
    test_set = int((train_set.pow(2).sum(1) < 2/pi)==True)
    return train_set, test_set

def create_model(nb_layers=3, layer_size=16):
    fc1 = Linear.Linear(2, layer_size)
    relu = ReLU.ReLU()
    
    tanh = Tanh.Tanh()
    layers_list = []
    for i in range(nb_layers):
        fc = Linear.Linear(layer_size, layer_size)
        layers_list.append(fc)
        layers_list.append(relu)
    fc2 = Linear.Linear(layer_size, 2)
    
    sequence = [fc1, relu] +
        layers_list +
        [fc2, tanh]
        
    return Sequential.Sequential(sequence)

"""
the following isn't adapted to the framework yet
"""

def train_model(model, train_input, train_target):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 250

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
def compute_nb_errors(model, data_input, data_target, batch_size = 100):
    nb_errors = 0
    for b in range(0, data_input.size(0), batch_size):
        target = data_target[b:b+batch_size]
        output = model(data_input[b:b+batch_size]).argmax(1)
        nb_errors += (target != output).sum()
    return nb_errors