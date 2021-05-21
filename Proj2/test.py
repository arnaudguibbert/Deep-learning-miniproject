from utils import train_model
import os 
from utils import generate_disc_set, create_model, assess_model
from pytorchNet import generate_contours
try:
    import pandas
    import seaborn
    pandas_flag = True
    print("Pandas and seaborn are installed graphs will be plotted")
except ModuleNotFoundError:
    pandas_flag = False
    print("Pandas or seaborn is not installed, to plot the graphs please install these librairies")

# Specify what you want
lineplot = True # Full evluation of the model (takes some time)
contour = True # Contour prediction of the model
compare_with_pytorch = False

# Specify the parameters
run = 2
#epochs to train the model
epochs = 10
#Epochs for ploting evolution of the results
epochs_contour = 100
granularity = 1

# Let the code do the rest
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
    print("Assess model performances \n")
    assess_model(create_model,epochs,granularity,
                 run,save_file="frw_evaluation",
                 save_data="frw_evaluation",pandas_flag=pandas_flag)

if contour:
    print("Generate countour \n")
    model = create_model()
    train_model(model,train_inputs,train_targets,epochs=epochs_contour)
    generate_contours(model,epochs=epochs_contour,save_fig="frw_contour")
