from metrics import Cross_validation, std_accuracy
from architecture import LugiaNet, oO_Net, BigNaive
try:
    import pandas
    import seaborn
    pandas_flag = True
    print("Pandas and seaborn are installed graphs will be plotted")
except ModuleNotFoundError:
    pandas_flag = False
    print("Pandas or seaborn is not installed, to plot the graphs please install these librairies")
import matplotlib.pyplot as plt
import torch
import os
import time

# Specify the parameters you want 
max_epochs = 10
granularity = 2
runs = 2

# True for loading pretrain and compute accuracy over a random test dataset, else it retrain the best hyperparameters
load_pretrain = False
best_hyper_Oonet = [2,False,[0.2, 0.8]]
best_hyper_Lugianet = [3]

#True if we want to retrain our model for multiple hyperameter
find_hyperparameters = False
#oO_Net hyperparameter : [embedded dimension of naive net,Use Resnet,[weight_loss]]
valid_Oo_args = None#[[4,False,[0.2, 0.8]],[4,True,[0.2, 0.8]]]
valid_Lugia_args = [[3]]

# Let the code do the rest
directories = ["figures","data_architectures"]

# Create directories if necessary 
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

if find_hyperparameters:
    # Validation of hyperparameters for LugiaNet
    if valid_Lugia_args is not None:

        print("Launch Lugia hyperparameter research \n")
        # Valid architectures
        valid_Lugia_architectures = [LugiaNet]*len(valid_Lugia_args)
        # Initialize the cross validation to determine the hyperparameters
        validation_Lugia = Cross_validation(valid_Lugia_architectures,
                                            valid_Lugia_args,
                                            epochs=max_epochs,
                                            steps=granularity,runs=runs,pandas_flag=pandas_flag)
        validation_Lugia.run_all(save_data="validation_Lugia",test=False)

        if pandas_flag:
            fig = plt.figure(figsize=[14,7])
            validation_Lugia.plot_std(fig,[1,1,1],test=False)
            fig.savefig("figures/boxplot_validation_LugiaNet.svg")

    if valid_Oo_args is not None:
        # Validation of hyperparameters for Oonet
        print("Launch Oo hyperparameter research \n")
        valid_Oo_architectures = [oO_Net]*len(valid_Oo_args)
        # Initialize the cross validation to determine the hyperparameters
        validation_Oo = Cross_validation(valid_Oo_architectures,
                                            valid_Oo_args,
                                            epochs=max_epochs,
                                            steps=granularity,runs=runs,pandas_flag=pandas_flag)
        validation_Oo.run_all(save_data="validation_Oo",test=False)

        if pandas_flag:
            fig = plt.figure(figsize=[14,7])
            validation_Oo.plot_std(fig,[1,1,1],test=False)
            fig.savefig("figures/boxplot_validation_OoNet.svg")

# Evaluate the performances on the test set
if (best_hyper_Oonet is not None and best_hyper_Lugianet is not None) or load_pretrain==True:
    if load_pretrain ==False :
        # Architectures to test on the test set 
        final_architectures = [oO_Net,BigNaive]
        # List of the best hyperparameters found so far
        final_args = [best_hyper_Oonet,[]]
        # Initialize the cross validation to determine the hyperparameters of some architectures
        Test_algo = Cross_validation(final_architectures,
                                    final_args,
                                    epochs=max_epochs,
                                    steps=granularity,runs=runs,pandas_flag=pandas_flag)

        Test_algo.run_all(test=True,save_data="_final_test")

        if pandas_flag:
            fig_test = plt.figure(figsize=[14,7])
            Test_algo.plot_evolution_all(fig_test,[1,2,1],type_perf=2)
            Test_algo.plot_std(fig_test,[1,2,2],test=True)
            fig_test.savefig("figures/test_set_final.svg")

        print("The final results are available in the figures folder")

        std_accuracy("data_architectures/accuracy_final_test.csv",
                     save_data="_final_metrics")

        print("The data acquired during the experiment is available data_architectures/folder")


    else :
        #Declaration of the model
        pretrained_oO_Net=oO_Net()
        pretrained_Lugia=LugiaNet(3)
        pretrained_BigNaive=BigNaive()

        #loading of the bias ad weight
        pretrained_oO_Net.load_state_dict(torch.load('model/oO_Net (4,False,[0.2, 0.8])_weights.pth'))
        pretrained_Lugia.load_state_dict(torch.load('model/LugiaNet (3)_weights.pth'))
        pretrained_BigNaive.load_state_dict(torch.load('model/BigNaive_weights.pth'))

        pretrained_oO_Net.eval()
        pretrained_Lugia.eval()
        pretrained_BigNaive.eval()

        #We call Cross_validation to use the accuracy and data retrieving function
        mysave = Cross_validation(architectures=[],args=[[]],pandas_flag=pandas_flag) 
        
        #load a test dataset
        _, _, _ ,test_input ,test_target ,test_classes=mysave.split_data()

        accuracy_oO_Net = mysave.accuracy(pretrained_oO_Net,test_input,test_target,test_classes)
        accuracy_Lugia = mysave.accuracy(pretrained_Lugia,test_input,test_target,test_classes)
        accuracy_BigNaive = mysave.accuracy(pretrained_BigNaive,test_input,test_target,test_classes)
        print("Obtained accuracy for this test dataset :\n")
        print(" oO_Net: {}\n Lugia : {}\n BigNaive :{}\n".format(accuracy_oO_Net,accuracy_Lugia,accuracy_BigNaive))
