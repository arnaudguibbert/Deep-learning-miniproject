# README project 1

This project aimed at testing different architectures to compare two digits, each corresponding to a channel of a 2x14x14 image.  
To that end, we built various architectures, trained them and tested them. 

## Getting started
Run test.py from the terminal.  
This will create folders **data_architecture** and **figure** containing results of oO Net and Lugia Net on the validation set.  
  
data_architecture contains three files:  
* corres_index.csv: maps an architecture and a hyperparmeter to an index (e.g. 3 <-- oO_Net, (4))
* accuracy.csv: maps an index (defined above) and a run number to an accuracy and a number of epochs
* time.csv: maps a run number and an index (defined above) to the time it took to one of our machine to generate the results.  
  
figure contains three files:
*boxplot_validation.svg: boxplot of results for various hyperparameters

  
## This repository contains:  
* **dlc_practical_prologue.py** contains the function *generate_pair_sets* which generates the desired number of train/target sets
* In **architecture.py**, we define all our models.
* **metrics.py** contains a method *train_model* which can train any of the models defined in architecture.py, regardless of the number of losses used in the model.  
It also contains the class *Cross_validation*, with various plots, counts, and tests to evaluate the performances.
* **test.py** saves .csv files containing results associated to specific architecture/parameters choice, and the corresponding .svg graphs
* **testers.ipynb** is a notebook with various results and plots
