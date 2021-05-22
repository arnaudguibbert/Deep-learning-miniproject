# Project 2

The objective of this project is to design a mini “deep learning framework” using only pytorch’s
tensor operations and the standard math library, hence in particular without using autograd or the
neural-network modules. 

## Getting started
Running the test.py file will allow you to reproduce the results exposed in th report, in particular it will train a multilayer perceptron on a toy data set. Feel free to change the parameters of the training by simply modifying the header of the test.py file. You will be able to follow the evolution of the run as test.py file displays a terminal printer. Running test.py file will create folders **data_architecture** and **figure** and respectively store into them the data acquired during the expriment and the graphs. WARNING The seaborn and pandas librairies are necessary to plot the graphs, make sure these librairies are installed, otherwise only the data will be available. 

data folder contains one file:  
* frw_evaluation.csv that contains the data acquired during the performances evaluation of the MLP
  
figures folder contains two type of files:
* frw_contour.svg Contour plot where the predictions of the MLP are plotted after training. 
* frw_evaluation.svg Evolution od the accuracy with respect to the number of epochs. 
* compare.avi Video that compares the boundaries found by our framework vs the ones found by pytorch
* classes.png Class diagram

## The repository contains:  
* **utils.py** generates the data, and contains methods to train and test our framework.
* **framework.py** contains our framework. We first define classes containing activation functions, loss functions, linear layers. Then, we define
a class 'Sequential' similar to the 'sequential' attribute of PyTorch. Finally we define the class FrameworkModule, basically empty but acting as a more general framework for potential enhancements of the framework.
* **pytorchNet.py** contains a PyTorch implementation of the same model as the one we used in test.py. This allows you to compare pytorch with the framework provided. 
