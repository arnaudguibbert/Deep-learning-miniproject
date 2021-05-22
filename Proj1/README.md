# README project 1

This project aimed at testing different architectures to compare two digits, each corresponding to a channel of a 2x14x14 image. To that end, One basic CNN (denoted BigNaive) and two more elaborated architectures have been developped. Their structures are detailed in the report. The elaborated architectures have some hyperparameters, then an hyperparameter search is first done on a validation set. Finally the performances of the three architectures (with the best hyper) are evaluated on a test set. 

## Getting started
The test.py file allows you to perform the hyperparameter search as well as evaluating the final performances on the test set. The provided form of the test.py allows you to reproduce the results exposed in the report, nevertheless you are free to change the parameters at the head of the test.py file. You will be able to follow the evolution of the run as test.py file displays a terminal printer. Running test.py file will create folders **data_architecture** and **figure** and respectively store into them the data acquired during the expriment and the graphs. WARNING The seaborn and pandas librairies are necessary to plot the graphs, make sure these librairies are installed. otherwise only the data will be available. 
  
data_architecture folder contains three type of files:  
* corres_index.csv: maps an architecture and a hyperparmeter to an index (e.g. 3 <-- oO_Net, (4))
* accuracy.csv: maps an index (defined above) and a run number to an accuracy, a number of epochs, and a type (0 for training, 1 for validation, 2 for test)
* time.csv: maps a run number and an index (defined above) to the time it took to one of our machine to generate the results.  
  
figures folder contains two type of files:
* boxplot_validation.svg: boxplot of results for various hyperparameters  
* test_set_final.svg: Performances of the best architectures on the test set
  
## This repository contains:  
* **dlc_practical_prologue.py** contains the function *generate_pair_sets* which generates the desired number of train/target sets.
* In **architecture.py**, we define all of our models.
* **metrics.py** contains a method *train_model* which can train any of the models defined in architecture.py, regardless of the number of losses used in the model.  
It also contains the class *Cross_validation*, this class allows you to train, perform hyperparameter search and compare the architectures, please see comments provided with that class.  
Two additional functions are provided *normalize* and *std_accuracy*, they are already well explained in the code. 
* **test.py** saves .csv files containing results associated to specific architecture/parameters choice, and the corresponding .svg graphs.
