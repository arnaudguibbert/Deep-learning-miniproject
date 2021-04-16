from Linear import *
from MSELoss import *
from RELu import *
from Sequential import *
from Tanh import *
from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch # à corriger

def generate_disc_set(nb=1000):
    train_set = torch.empty(nb, 2).uniform_()
    test_set = train_set - torch.empty(1).fill_(0.5)
    test_set = (test_set.pow(2).sum(1) < 1/sqrt(2*pi)) # Outside the circle is actually quite rare (about 2.4% of the samples)
    test_set = torch.Tensor([int(x) for x in test_set]) # à corriger
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
    
    sequence = [fc1, relu] + layers_list + [fc2, tanh]
        
    return Sequential.Sequential(sequence)


def compute_nb_errors(model, data_input, data_target, batch_size = 100):
    nb_errors = 0
    for b in range(0, data_input.size(0), batch_size):
        target = data_target[b:b+batch_size]
        
        # need something like "with torch.no_grad" here
        
        output = model.forward(data_input[b:b+batch_size]).argmax(1)
        nb_errors += (target != output).sum()
    return nb_errors


def train_model(model, train_input, train_target, 
                nb_epochs = 100, mini_batch_size = 100, lr = 5e-2,
               create_plot=False):
    # still need to add the figure
    
    model.reset()
    
    if create_plot:
        errors = []
    
    for e in range(nb_epochs):
        if create_plot:
            error = compute_nb_errors(model, train_input, train_target)
        
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model.forward((train_input.narrow(0, b, mini_batch_size)))
            loss = MSELoss.forward(train_target, output)
            grdwrtoutput = MSELoss.backward()
            model.zero_grad()
            model.backward(grdwrtoutput)
            model.optimization_step(lr)
    
    
    
    
    
    
    