import os 
from utils import generate_disc_set, create_model, assess_model, train_model
from pytorchNet import generate_contours
try:
    import pandas as pd
    import seaborn as sns
    pandas_flag = True
except ModuleNotFoundError:
    pandas_flag = False

# Specify what you want
lineplot = True # Full evluation of the model (takes some time)
contour = True # Contour prediction of the model

# Specify the parameters
run = 2
#epochs to train the model
epochs = 50
#Epochs for ploting evolution of the results
epochs_contour = 100
granularity = 1

# Let the code do the rest Do not change anything in the rest of the code

print("################### PARAMETERS ################### \n")

if not pandas_flag:
    print("Accuracy evolution graph : False (pandas or seaborn not installed, please install these librairies to generate the accuracy evolution graph, only the data will be saved)")
else:
    print("Accuracy evolution graph : ",lineplot)
print("Boundary graph : ",contour)
print("Number of runs : ",run)
print("Epochs for boundary graph : ",epochs_contour)

print("\nThese parameters can be easily modified in the header of the test.py file \n")

directories = ["figures","data"]

# Create directories if necessary 
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

# Generate data set
inputs, targets = generate_disc_set(2000)
test_inputs, test_targets = inputs[:1000], targets[:1000]
train_inputs, train_targets = inputs[1000:], targets[1000:]

if pandas_flag:
    sns.set_style("darkgrid")

if lineplot:
    print("\n################### Assess model performances (accuracy evolution) ###################\n")
    assess_model(create_model,epochs,granularity,
                 run,save_file="frw_evaluation",
                 save_data="frw_evaluation",pandas_flag=pandas_flag)
    print("\n################### Data saved in data folder, and graphs in figure folder ###################\n")

if contour:
    print("\n################### Boundary graph ###################\n")
    model = create_model()
    train_model(model,train_inputs,train_targets,epochs=epochs_contour)
    generate_contours(model,epochs=epochs_contour,save_file="frw_contour")
    print("\n################### Data saved in data folder, and graphs in figure folder ###################\n")
