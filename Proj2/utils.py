from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from torch import empty
import torch
import framework as frw

def generate_disc_set(nb=1000):
    train_set = empty(nb, 2).uniform_()
    train_target = train_set - empty(1).fill_(0.5)
    train_target = (train_target.pow(2).sum(dim=1) < 1/(2*pi))*1
    train_target[train_target == 0] = -1
    return train_set, train_target.view(-1,1)

def create_model(nb_layers=3, layer_size=16):
    model = frw.Sequential(
        frw.Linear(2, layer_size),
        frw.ReLU(),
        frw.Linear(layer_size, layer_size),
        frw.ReLU(),
        frw.Linear(layer_size, layer_size),
        frw.ReLU(),
        frw.Linear(layer_size, layer_size),
        frw.ReLU(),
        frw.Linear(layer_size, 1),
        frw.Tanh()
    )
    return model

def compute_nb_errors(model, data_input, data_target):
    output = model.forward(data_input, no_grad=True)
    output[output < 0] = -1
    output[output >= 0] = 1
    nb_errors = ((data_target != output)*1).sum()
    accuracy = (1 - nb_errors/data_input.shape[0])*100
    return accuracy.item()

def assess_model(model_gen,epochs,granularity,runs=10):
    row_format = '{:<20}{:<25}{:<25}'
    header = ["Run","Accuracy Train","Accuracy Test"]
    under_header = ["-"*len(word) for word in header]
    print(row_format.format(*header)) # Print the header
    print(row_format.format(*under_header)) # Print the the under_header
    np_data = np.empty((0,4))
    for run in range(runs):
        model = model_gen()
        inputs, targets = generate_disc_set(2000)
        test_inputs, test_targets = inputs[:1000], targets[:1000]
        train_inputs, train_targets = inputs[1000:], targets[1000:]
        accu_train = compute_nb_errors(model,train_inputs, train_targets)
        accu_test = compute_nb_errors(model,test_inputs, test_targets)
        row_train = np.array([run,accu_train,epoch,0]).reshape(1,-1)
        row_test = np.array([run,accu_test,epoch,1]).reshape(1,-1)
        np_data = np.concatenate((np_data,row_train,row_test),axis=0)
        for epoch in range(0,epochs,granularity):
            train_model(model,train_inputs,train_targets)
            accu_train = compute_nb_errors(model,train_inputs, train_targets)
            accu_test = compute_nb_errors(model,test_inputs, test_targets)
            row_train = np.array([run,accu_train,epoch,0]).reshape(1,-1)
            row_test = np.array([run,accu_test,epoch,1]).reshape(1,-1)
            np_data = np.concatenate((np_data,row_train,row_test),axis=0)
        row = row.reshape(-1).tolist()
        print(row_format.format(*row))
    columns = ["Run","Accuracy","Epochs","type"]
    data_pd = pd.DataFrame(np_data,columns=columns)
    fig = plt.figure(figsize=[10,6])
    ax = fig.add_subplot(1,1,1)
    sns.set_style("darkgrid")
    sns.lineplot(data=data_pd,x="Epochs",y="Accuracy train",ax=ax)
    sns.lineplot(data=data_pd,x="Epochs",y="Accuracy test",ax=ax)
    plt.show()
    return np_data

def train_model(model,train_inputs,train_targets,
                epochs=100,mini_batch_size=100,lr=5e-2):
    optimizer = frw.Optim(model,lr=lr)
    criterion = frw.MSELoss()
    for _ in range(epochs):
        for b in range(0,train_inputs.shape[0],mini_batch_size):
            inputs = train_inputs.narrow(0, b, mini_batch_size)
            output = model.forward(inputs)
            target = train_targets.narrow(0, b, mini_batch_size)
            loss = criterion.forward(output,target)
            grdwrtoutput= criterion.backward()
            model.zero_grad()
            model.backward(grdwrtoutput)
            optimizer.optimize()