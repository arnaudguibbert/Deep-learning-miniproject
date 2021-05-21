from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ModuleNotFoundError:
    pass
try:
    import pandas as pd
except ModuleNotFoundError:
    pass  
from torch import empty
import framework as frw

def generate_disc_set(nb=1000):
    """
    Goal:
    Generate the inputs and labels of a data set. The inputs are two dimensional
    and included in [0,1]x[0,1]. If the input is in the circle of radius r = 1/(2*pi)^(0.5)
    and center [0.5,0.5] then the label is equal to 1, otherwise.
    Inputs:
    nb = int - number of data points to be generated
    Outputs:
    inputs = torch tensor - Nx2 (N number of data points)
    targets = torch.tensor - Nx1 (N number of data points)
    """
    inputs = empty(nb, 2).uniform_(0,1) 
    centered = inputs - 0.5
    labels = (centered.pow(2).sum(dim=1) < 1/(2*pi))*1
    labels[labels == 0] = -1
    return inputs, labels.view(-1,1)

def create_model(layer_size=16):
    """
    Goal:
    Create a MLP with 3 hidden layers using the framework created
    Inputs:
    layer_size = int - number of hidden units
    Outputs:
    model = Sequential framework object - MLP
    """
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
    """
    Goal:
    Given a model compute the accuracy of this model on a given set. 
    Inputs:
    model = Framework object, could be a sequential object for instance - model to be evaluated
    data_input = torch tensor - Nx2 (N number of data points)
    data_target = torch.tensor - Nx1 (N number of data points)
    Outputs:
    accuracy = float - accuracy of the model on the set
    """
    output = model.forward(data_input, no_grad=True) # Output of the model
    output[output < 0] = -1 # Convert to labels
    output[output >= 0] = 1
    nb_errors = ((data_target != output)*1).sum()
    accuracy = (1 - nb_errors/data_input.shape[0])*100
    return accuracy.item()

def train_model(model,train_inputs,train_targets,
                epochs=100,batch_size=50,lr=1e-1):
    """
    Goal:
    Train a model generated with the framework
    Inputs:
    model = Framework object, could be a sequential object for instance - model to be trained
    train_inputs = torch tensor - Nx2 (N number of data points)
    train_targets = torch.tensor - Nx1 (N number of data points)
    epochs = int - number of epochs for the training
    batch_size = int - batch size for the SGD
    lr = float - learning rate for the SGD
    Outputs:
    """
    optimizer = frw.Optim(model,lr=lr) # Define the optimizer
    criterion = frw.MSELoss() # Define the loss
    for _ in range(epochs):
        for b in range(0,train_inputs.shape[0],batch_size):
            inputs = train_inputs.narrow(0, b, batch_size)
            output = model.forward(inputs)
            target = train_targets.narrow(0, b, batch_size)
            loss = criterion.forward(output,target) # Compute the loss
            grdwrtoutput= criterion.backward() # Compute the loss gradient
            model.zero_grad() # Reset the gradients to 0
            model.backward(grdwrtoutput) # Compute the new gradients
            optimizer.optimize() # Optimization step

def assess_model(model_gen,epochs,
                 granularity,
                 runs=10,
                 save_file=None,
                 save_data=None,
                 pandas_flag=False):
    """
    Goal:
    Assess the performances of an architecture. Evaluate the performances on several data sets
    Plot a graph of the accuracy wand its standard deviation with respect to the epochs.
    Inputs:
    model_gen = function or class that returned a model when it is called
    epochs = int - number of epochs for the training step
    granularity = int - granularity of the graph
    runs = int - how many times the model is trained
    save_file = string - name of the file for the graph
    save_file = string - name of the file for the data
    Outputs:
    """
    # Graph parameters
    plt.rc('axes', titlesize=16)   
    plt.rc('axes', labelsize=14)
    plt.rc('legend', fontsize=13)   
    plt.rc('legend', title_fontsize=13)   
    row_format = '{:<20}{:<25}{:<25}' # Row format for the logs 
    header = ["Run","Accuracy Train","Accuracy Test"] 
    under_header = ["-"*len(word) for word in header]
    print(row_format.format(*header)) # Print the header
    print(row_format.format(*under_header)) # Print the the under_header
    np_data = np.empty((0,4)) # Data will be stored there
    for run in range(runs): # Evaluate the model runs times
        model = model_gen() # Generate a model
        inputs, targets = generate_disc_set(2000)
        # Split train/test
        test_inputs, test_targets = inputs[:1000], targets[:1000]
        train_inputs, train_targets = inputs[1000:], targets[1000:]
        # Compute the initial accuracy
        accu_train = compute_nb_errors(model,train_inputs, train_targets)
        accu_test = compute_nb_errors(model,test_inputs, test_targets)
        row_train = np.array([run,accu_train,0,0]).reshape(1,-1)
        row_test = np.array([run,accu_test,0,1]).reshape(1,-1)
        np_data = np.concatenate((np_data,row_train,row_test),axis=0)
        for epoch in range(0,epochs,granularity):
            # Train succesively the model
            train_model(model,train_inputs,train_targets)
            accu_train = compute_nb_errors(model,train_inputs, train_targets)
            accu_test = compute_nb_errors(model,test_inputs, test_targets)
            row_train = np.array([run,accu_train,epoch,0]).reshape(1,-1)
            row_test = np.array([run,accu_test,epoch,1]).reshape(1,-1)
            np_data = np.concatenate((np_data,row_train,row_test),axis=0)
        row = [run,round(row_train[0,1],2),
               round(row_test[0,1],2)]
        print(row_format.format(*row))
    # Convert into a pandas data set
    columns = ["Run","Accuracy","Epochs","type"]
    if pandas_flag:
        data_pd = pd.DataFrame(np_data,columns=columns)
        # Plot the graph
        fig = plt.figure(figsize=[10,5])
        ax = fig.add_subplot(1,1,1)
        sns.set_style("darkgrid")
        sns.lineplot(data=data_pd,x="Epochs",y="Accuracy",ax=ax,hue="type")
        ax.set_title("Accuracy evolution",fontsize=14)
        handles, labels = ax.get_legend_handles_labels()
        labels = [(i == "0.0")*"train" + (i == "1.0")*"test" for i in labels]
        ax.legend(handles,labels,title="Set",loc="lower right")
    # Save the files 
        if save_file is not None:
            fig.savefig("figures/" + save_file + ".svg",dpi=200)
    if save_data is not None:
        name_file = "data/" + save_data +".csv"
        if pandas_flag:
            data_pd.to_csv(name_file)
        else:
            np.savetxt(name_file,np_data,
                       delimiter=",",
                       header=",".join(columns))
