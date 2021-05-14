from metrics import Cross_validation
from architecture import LugiaNet, oO_Net, BigNaive
import matplotlib.pyplot as plt
import os
import time

max_epochs = 30
granularity = 2
runs = 15
find_hyperparameters = True
best_hyper_Oonet = None
best_hyper_Lugianet = None
valid_Oo_args = [[2],[3],[4],[5]]
valid_Lugia_args = [[2],[3],[4],[5]]

directories = ["figures","data_architectures"]

for directory in directories:
    if not os.path.exists(directory):
        os.mkdir(directory)

if find_hyperparameters:
    print("Launch Lugia hyperparameter research \n")
    # Valid architectures
    valid_Lugia_architectures = [LugiaNet]*len(valid_Lugia_args)
    # Initialize the cross validation to determine the hyperparameters of some architectures
    validation_Lugia = Cross_validation(valid_Lugia_architectures,
                                        valid_Lugia_args,
                                        epochs=max_epochs,
                                        steps=granularity,runs=runs)
    validation_Lugia.run_all(save_data="validation_Lugia",test=False)

    fig = plt.figure(figsize=[14,7])
    validation_Lugia.plot_std(fig,[1,1,1],test=False)
    fig.savefig("figures/boxplot_validation_LugiaNet.svg")

    print("Launch Oo hyperparameter research \n")
    valid_Oo_architectures = [LugiaNet]*len(valid_Oo_args)
    # Initialize the cross validation to determine the hyperparameters of some architectures
    validation_Oo = Cross_validation(valid_Oo_architectures,
                                     valid_Oo_args,
                                     epochs=max_epochs,
                                     steps=granularity,runs=runs)
    validation_Oo.run_all(save_data="validation_Oo",test=False)

    fig = plt.figure(figsize=[14,7])
    validation_Oo.plot_std(fig,[1,1,1],test=False)
    fig.savefig("figures/boxplot_validation_OoNet.svg")


if best_hyper_Oonet is not None and best_hyper_Lugianet is not None:
    # Architectures to test on the test set 
    final_architectures = [LugiaNet,oO_Net,BigNaive]
    # List of the best hyperparameters found so far
    final_args = [best_hyper_Lugianet,best_hyper_Oonet,[]]
    # Initialize the cross validation to determine the hyperparameters of some architectures
    Test_algo = Cross_validation(final_architectures,
                                 final_args,
                                 epochs=max_epochs,
                                 steps=granularity,runs=runs)

    Test_algo.run_all(test=True)

    fig_test = plt.figure(figsize=[14,7])
    Test_algo.plot_evolution_all(fig,[1,2,1],type_perf=2)
    Test_algo.plot_std(fig,[1,2,2],test=True)
    fig_test.savefig("figures/test_set_final.svg")

    print("The final results are available in the figures folder")