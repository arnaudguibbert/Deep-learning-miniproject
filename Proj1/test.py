from metrics import Cross_validation, std_accuracy
from architecture import LugiaNet, oO_Net, BigNaive
try:
    import pandas
    import seaborn
    pandas_flag = True
except ModuleNotFoundError:
    pandas_flag = False
import matplotlib.pyplot as plt
import torch
import os
import time

# Specify the parameters you want 
max_epochs = 10
granularity = 2
runs = 5

# True for loading pretrain and compute accuracy over a random test dataset, else it retrain the best hyperparameters
load_pretrain = False
best_hyper_Oonet = [1,2,False,[0.2, 0.8]]
best_hyper_Lugianet = [1]

#True if we want to retrain our model for multiple hyperameter
find_hyperparameters = False
#oO_Net hyperparameter : [embedded dimension of naive net,Use Resnet,[weight_loss]]
valid_Oo_args = [[1,4,True,[0.2, 0.8]]]
valid_Lugia_args = [[1]]

# Let the code do the rest Do not change anything in the rest of the code

print("################### PARAMETERS ################### \n")

if not pandas_flag:
    print("Graph plot : False (pandas or seaborn not installed, please install these librairies to generate the accuracy evolution graph, only the data will be saved)")
else:
    print("Graph plot : ",True)
print("Use pretained models : ",load_pretrain)
print("Hyperparameter search : ",find_hyperparameters)
if find_hyperparameters:
    head_lugia = "---------- Lugia hyperparameters set ----------"
    head_Oo = "---------- Oo hyperparameters set ----------"
    print(head_lugia)
    for hyper in valid_Lugia_args:
        print(hyper)
    print(head_Oo)
    for hyper in valid_Oo_args:
        print(hyper)
    print("-"*len(head_lugia))
print("Final test with the best architectures : ",(best_hyper_Oonet is not None and best_hyper_Lugianet is not None) or load_pretrain==True)
print("Number of runs : ",runs)
print("Epochs : ",max_epochs)
print("\nThese parameters can be easily modified in the header of the test.py file \n")

directories = ["figures","data_architectures"]

# Create directories if necessary 
for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

if find_hyperparameters:
    # Validation of hyperparameters for LugiaNet
    if valid_Lugia_args is not None:

        print("\n################### Launch Lugia hyperparameter search ###################\n")
        # Valid architectures
        valid_Lugia_architectures = [LugiaNet]*len(valid_Lugia_args)
        # Initialize the cross validation to determine the hyperparameters
        validation_Lugia = Cross_validation(valid_Lugia_architectures,
                                            valid_Lugia_args,
                                            epochs=max_epochs,
                                            steps=granularity,runs=runs,pandas_flag=pandas_flag)
        validation_Lugia.run_all(save_data="_validation_Lugia",test=False)

        if pandas_flag:
            fig = plt.figure(figsize=[14,7])
            validation_Lugia.plot_std(fig,[1,1,1],test=False)
            fig.savefig("figures/boxplot_validation_LugiaNet.svg")

    if valid_Oo_args is not None:
        # Validation of hyperparameters for Oonet
        print("\n################### Launch Oo hyperparameter search ###################\n")
        valid_Oo_architectures = [oO_Net]*len(valid_Oo_args)
        # Initialize the cross validation to determine the hyperparameters
        validation_Oo = Cross_validation(valid_Oo_architectures,
                                            valid_Oo_args,
                                            epochs=max_epochs,
                                            steps=granularity,runs=runs,pandas_flag=pandas_flag)
        validation_Oo.run_all(save_data="_validation_Oo",test=False)

        if pandas_flag:
            fig = plt.figure(figsize=[14,7])
            validation_Oo.plot_std(fig,[1,1,1],test=False)
            fig.savefig("figures/boxplot_validation_OoNet.svg")


# Evaluate the performances on the test set
if (best_hyper_Oonet is not None and best_hyper_Lugianet is not None) or load_pretrain==True:
    if not load_pretrain :

        print("\n################### Launch assessment of the best architectures on the final test set ###################\n")
        # Architectures to test on the test set 
        final_architectures = [oO_Net,LugiaNet,BigNaive]
        # List of the best hyperparameters found so far
        final_args = [best_hyper_Oonet,best_hyper_Lugianet,[]]
        # Initialize the cross validation to determine the hyperparameters of some architectures
        Test_algo = Cross_validation(final_architectures,
                                     final_args,
                                     epochs=max_epochs,
                                     steps=granularity,runs=runs,pandas_flag=pandas_flag)

        Test_algo.run_all(test=True,save_data="_final_test")

        if pandas_flag:

            fig_test = plt.figure(figsize=[20,8])
            Test_algo.plot_evolution_all(fig_test,[1,5,(1,3)],type_perf=2)
            Test_algo.plot_std(fig_test,[1,5,(4,5)],test=True)
            plt.subplots_adjust(wspace=0.5)
            fig_test.savefig("figures/test_set_final.svg")

            print("\n################### The final results are available in the figures folder ###################\n")

        print("\n################### Sum up of the results obtained ###################\n")

        std_accuracy("data_architectures/accuracy_final_test.csv",archi_names=Test_algo.archi_names,
                     save_data="_final")

        print("\n################### The data acquired during the experiment is available data_architectures/folder ###################\n")


    else :

        print("\n################### Use pertrain models ###################\n")
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
