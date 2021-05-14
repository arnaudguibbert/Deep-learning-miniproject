import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import numpy as np
from time import perf_counter
from dlc_practical_prologue import generate_pair_sets
import torchvision.transforms as transform


def train_model(model, train_input, train_target, train_classes,
                nb_epochs=50, 
                mini_batch_size = 100, 
                eta = 0.05):
    """
    Goal:
    Train a given model 
    Inputs:
    model = nn.Module class object - model you want to train
    train_input = tensor - size (Nx2x14x14) (N number of samples)
                  input of the training datas
    train_target = tensor - size (N) (N number of samples)
                   targets - belongs to {0,1}
    train_classes = tensor - size (Nx2) (N number of samples)
                    Classes (i.e. numbers) of the two images - belongs to {1,...,10}
    nb_epochs = int - Number of epochs for the training
    mini_batch_size = int - size of the mini batch size
    eta = float - learning rate 
    Outputs:
    """
    criterion = nn.CrossEntropyLoss() # Define the loss
    optimizer = torch.optim.Adam(model.parameters(), lr = eta) # Define the optimizer
    # For each output of the model a target type is associated (auxiliary losses)
    target_type = model.target_type 
    # For each loss a weight is defined
    weights_loss = model.weights_loss
    for epochs in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            # Compute the output of the model
            output = model(train_input.narrow(0, b, mini_batch_size))
            # If there is multiple outputs (auxiliary losses)
            if len(target_type) > 1:
                loss_list = [] # List where the losses will be stored
                for i,target in enumerate(target_type):
                    # Target 0 means that the output predicts whether the first number 
                    # is higher or lower than the second one
                    if target == "target0":
                        # Compute the auxiliary loss
                        aux_loss = criterion(output[i], 
                                             train_target.narrow(0, b, mini_batch_size))
                    # Target 1 means that the output predicts the classes of the images
                    elif target == "target1":
                        # Compute the auxiliary loss
                        aux_loss = criterion(output[i], 
                                             train_classes.narrow(0, b, mini_batch_size))
                    else:
                        # print Error message
                        return "Unexpected value in the attribute target_type"
                    # Add the weighted auxiliary loss to the loss list
                    loss_list.append(aux_loss*weights_loss[i])
                # Sum the auxiliary losses
                loss = sum(loss_list)
            else:
                # Else if one output : no auxiliary losses
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad() # Reset the gradient tensors to 0
            loss.backward() # Perform a backward step
            optimizer.step() # Update the weights


class Cross_validation():

    def __init__(self,
                 architectures,
                 args,
                 steps=5,
                 runs=10,load=5000,epochs=50):
        """
        Goal:
        Inputs:
        architectures = list of class generating the architectures
        args = list of the arguments of each class
        steps = int - granularity for the graphs ()
        runs = int - number of times to retrain a new model
        load = int - number of samples you are loading (only 1000 will be used for train and test at each run)
        epochs = int - number of epochs for the training
        Outputs:
        """
        self.architectures = architectures # Get the list of architectures
        self.archi_names = [] # Store the name of the architectures
        for i,archi in enumerate(architectures):
            name = archi.__name__
            if len(args[i]) > 0:
                name += " ("
                for arg in args[i]:
                    name += str(arg) + ","
                name = name[:-1]
                name += ")"
            self.archi_names.append(name)
        self.args = args # Get the arguments for each architecture
        self.runs = runs # Number of runs
        # Columns of the data frame where all the data will be stored (will be used for the graphs)
        self.columns = ["run_id","architecture","accuracy","type","epochs"]
        # Create the data frames
        self.columns_time = ["architecture","time","run_id"]
        self.parameters_count = self.count_params()
        self.datatime = pd.DataFrame([[1e20,1e20,1e20]],columns=self.columns_time)
        self.dataframe = pd.DataFrame([[1e20,1e20,1e20,1e20,1e20]],columns=self.columns)
        # Load the the data set
        data = generate_pair_sets(load)
        self.size = 1000 # Number of samples used for training and testing at each run
        self.epochs = epochs # get the number of epochs
        # To be checked (are these two lines necessary ??)
        self.train_input, self.train_target, self.train_classes = data[0], data[1], data[2]
        self.test_input, self.test_target, self.test_classes = data[3], data[4], data[5]
        self.steps = steps # Get Granularity for the graphs
        self.ratio_valid = 0.4
        # Row format for the logs
        self.row_format = '{:<20}{:<15}{:<25}{:<25}{:<15}' # Define the display format
        #store it to see plot where the model fail, random initialisation
        self.errors_img = torch.empty(0,1) # torch.empty() marche aussi ?
        self.errors_target = torch.empty(0,1)
        self.errors_numbers = torch.empty(0,1)
        self.right_target = torch.empty(0,1)

    def count_params(self,save_data=None):
        """
        Goal:
        Count the number of parameters to be trained for each model. 
        Inputs:
        Outputs:
        param_count = list of size = number of architectures
                      Store the number of parameters for each model
        """
        param_count = []
        for i,archi in enumerate(self.architectures):
            model = archi(*self.args[i])
            n_params = 0
            with torch.no_grad():
                for params in model.parameters():
                    n_params += params.numel()
            param_count.append(n_params)
        if save_data is not None:
            data_param = np.array(self.archi_names + param_count).T
            param_pd = pd.DataFrame(data_param,columns=["Architectures","Number of parameters"])
            param_pd.to_csv("data_architectures/param_count" + save_data + ".csv",index=False)
        return param_count

    def data_augmentation(self,train_input):
        """
        Augment the data by :Random tilt between [-10,10] degree and RandomErasing 
        p – probability that the random erasing operation will be performed.
        scale – range of proportion of erased area against input image.
        ratio – range of aspect ratio of erased area.
        """
        rotation=transform.RandomRotation((-10,10))
        erasing=transform.RandomErasing(p=0.5, scale=(0.015, 0.015), ratio=(1, 1))
        tilted_train=rotation(train_input)
        train_input_augmented=erasing(tilted_train)
        return tilted_train

    def split_data(self,nb_classes=10,test=False):
        """
        To be modified to take into account the classes
        Goal:
        Extract randomly 1000 (size attribute) training and testing data points
        Inputs:
        Outputs:
        train_input = tensor - size (1000x2x14x14)
                      input of the training datas
        train_target = tensor - size (1000)
                       targets - belongs to {0,1}
        train_classes = tensor - size (1000x2)
                        Classes (i.e. numbers) of the two images - belongs to {1,...,10}
        validation_input = tensor - size (1000x2x14x14)
                      input of the validation datas
        validation_target = tensor - size (1000)
                       targets - belongs to {0,1}
        validation_classes = tensor - size (1000x2)
                        Classes (i.e. numbers) of the two images - belongs to {1,...,10}
        """
        shuffle = torch.randperm(self.train_input.shape[0])
        index_train = shuffle[:self.size]
        train_input = self.train_input[index_train]
        train_target = self.train_target[index_train]
        train_classes = self.train_classes[index_train]
        if not test:
            index_test = shuffle[-round(self.ratio_valid*self.size):]
            test_input = self.train_input[index_test]
            test_target = self.train_target[index_test]
            test_classes = self.train_classes[index_test]
        else:
            index_test = index_train
            test_input = self.test_input[index_test]
            test_target = self.test_target[index_test]
            test_classes = self.test_classes[index_test]
        return train_input, train_target, train_classes ,test_input ,test_target ,test_classes

    def accuracy(self,model,input,target,target_classes):
        """
        Goal:
        Compute the accuracy of a model on a given data set
        Inputs:
        input = tensor - size (Nx2x14x14) N number of samples
                input of the model
        target = tensor - size (N) N number of samples
                 targets - belongs to {0,1}
        target_classes = tensor - size (Nx2)
        Outputs:
        accuracy = float - Accuracy in percentage 
        """
        model.eval()
        with torch.no_grad(): # Shut down the autograd machinery
            output = model(input) # Compute the output of the model
            # If the model has auxiliary output
            if len(model.target_type) > 1: 
                 # Let's take the real one (by convention the first returned)
                main_output = output[0] 
            else:
                main_output = output
            _,predicted = torch.max(main_output,dim=1) # Compute the prediction
            #compute the error matrix
            errors_matrix = torch.where(target != predicted,1,0)
            # Compute the number of errors
            total_errors = errors_matrix.sum().item()
            #store the wrong set of image
            errors_index = torch.empty(0,1)
            errors_index = ((errors_matrix == 1).nonzero(as_tuple=True)[0])
            self.errors_img = input[errors_index]
            self.errors_target = predicted[errors_index]
            if len(model.target_type) > 1 and len(errors_index) != 0: 
                #We can see the errors only if the model has it as output
                self.errors_numbers=torch.argmax(output[1][errors_index],dim=1)
                self.right_target=target_classes[errors_index]
            # Compute the accuracy
            accuracy = (1 - total_errors/(target.shape[0]))*100
        model.train()
        return accuracy

    def get_errors(self):
        return self.errors_img, self.errors_target, self.errors_numbers

    def run_one(self,archi_name,test=False):
        """
        Goal:
        Train a model generated by the architecture "archi_name" "self.runs" times.
        Store the performances of these models in the data frame
        the performances are recorded each "self.step" epochs
        Inputs:
        archi_name = name of the architecture (i.e. name of the class)
        test = Boolean - are you validating hyperparameters or perform a final test 
               on the testing data
        Outputs:
        """
        if test:
            type_perf = 2
        else:
            type_perf = 1
        # Check that the name of the architectures is defined
        if not archi_name in self.archi_names:
            return "Unexpected value for archi_name"
        # Get the index of the architecture
        index = self.archi_names.index(archi_name)
        # Get the class 
        Myclass = self.architectures[index]
        # Get the arguments
        args = self.args[index]
        # The data to add to the data frame will be stored in this tensor
        new_data_time = torch.zeros(len(self.columns_time)).view(1,-1)
        new_data = torch.zeros(len(self.columns)).view(1,-1)
        for runs in range(self.runs): # Repeat runs times
            # Get a random data set of 1000 samples for training, same for testing 
            data = self.split_data(test=test)
            # Extract this
            train_input, train_target, train_classes = data[0], data[1], data[2]
            test_input, test_target, test_classes = data[3], data[4], data[5]
            # Create the model
            model = Myclass(*args)
            # Compute the initial accuracy 
            accuracy_train = self.accuracy(model,train_input,train_target,train_classes)
            accuracy_test = self.accuracy(model,test_input,test_target,test_classes)
            # Store it into the new_data tensor
            row_test = torch.tensor([runs,index,accuracy_test,type_perf,0]).view(1,-1)
            row_train = torch.tensor([runs,index,accuracy_train,0,0]).view(1,-1)
            new_data = torch.cat((new_data,row_train,row_test),dim=0)
            # Train the model and record the accuracy each self.steps epochs
            start = perf_counter() # Start the chrono
            for step in range(self.steps,self.epochs,self.steps):
                # Train the model for self.steps epochs
                train_model(model, 
                            train_input, 
                            train_target, 
                            train_classes, 
                            nb_epochs=self.steps)
                # Compute the accuracy on the train and test set
                accuracy_train = self.accuracy(model,train_input,train_target,train_classes)
                accuracy_test = self.accuracy(model,test_input,test_target,test_classes)
                # Store is into the new_data tensor
                row_test = torch.tensor([runs,index,accuracy_test,type_perf,step]).view(1,-1)
                row_train = torch.tensor([runs,index,accuracy_train,0,step]).view(1,-1)
                new_data = torch.cat((new_data,row_train,row_test),dim=0)
            # Store into the new_data_time tensor
            end = perf_counter() # Stop the chrono
            elapsed = (end - start) # Compute the elapsed time
            row_time = torch.tensor([index,elapsed,runs]).view(1,-1)
            new_data_time = torch.cat((new_data_time,row_time),dim=0)
            # Row to be displayed/logged
            row = [archi_name,runs,
                   round(accuracy_train,1),
                   round(accuracy_test,1),round(elapsed,1)]
            # Print a message about the performances of the architecture
            print(self.row_format.format(*row))
        # Remove the first artificial line
        new_data = new_data[1:]
        new_data_time = new_data_time[1:]
        # Add the new data to the existing data frame
        df = pd.DataFrame(data=new_data.tolist(),columns=self.columns)
        self.dataframe = self.dataframe.append(df,ignore_index=True)
        # Remove the first artificial line of the data frame
        df_time = pd.DataFrame(data=new_data_time.tolist(),columns=self.columns_time)
        self.datatime = self.datatime.append(df_time,ignore_index=True)
        self.remove_line()

    def run_all(self,test=False,save_data=None):
        """
        Goal:
        For each architecture : 
        Train a model generated by the architecture "self.runs" times.
        Store the performances of these models in the data frame
        the performances are recorded each "self.step" epochs
        Inputs:
        Outputs:
        """
        # Header to be displayed
        if test:
            accu_header = "Accuracy Test"
        else:
            accu_header = "Accuracy Validation"
        header = ["Architecture","Runs","Accuracy Train",accu_header,"Time"]
        under_header = ["-"*len(word) for word in header]
        print(self.row_format.format(*header)) # Print the header
        print(self.row_format.format(*under_header)) # Print the the under_header
        # For each architecture
        for archi_name in self.archi_names:
            self.run_one(archi_name,test=test)
        if save_data is not None:
            corres_pd = pd.DataFrame(self.archi_names,columns=["Architecture name"])
            corres_pd.to_csv("data_architectures/corres_index" + save_data + ".csv")
            self.dataframe.to_csv("data_architectures/accuracy" + save_data + ".csv",index=False)
            self.datatime.to_csv("data_architectures/time" + save_data + ".csv",index=False)

    def remove_line(self):
        """
        Goal:
        Remove the artficial first line of the data frame
        Inputs:
        Outputs:
        """
        # Remove the first line where accuracy was set to 1e20
        self.dataframe = self.dataframe.query("accuracy < 1e3")
        self.datatime = self.datatime.query("time < 1e10")

    def reset(self):
        """
        Goal:
        Reset the data frame (warning it will erase its content)
        Inputs:
        Outputs:
        """
        # Reset the data frame
        self.dataframe = pd.DataFrame([[1e20,1e20,1e20,1e20,1e20]],columns=self.columns)
        self.datatime = pd.DataFrame([[1e20,1e20,1e20]],columns=self.columns_time)
    
    def plot_std(self,figure,subplot,test=False):
        """
        Goal:
        Boxplot - Plot the standard deviations of the performances of each
        architectures on the training and testing set after having been trained
        for self.epochs epochs
        Inputs:
        figure = matplotlib figure - figure where the boxplot will be plotted
        subplot = list of size 3 - location of the boxplot in the figure
        Outputs:
        """
        # Set the style
        sns.set_style("darkgrid")
        # Add a subplot in the figure
        ax = figure.add_subplot(*subplot)
        # Title to be displayed 
        title = "Results (epochs = " + str(self.epochs) + ")"
        # Get the maximum number of epochs
        max_epochs = self.dataframe["epochs"].max()
        # Get the performances after being trained with max_epochs
        std_data = self.dataframe.query("epochs == " + str(max_epochs))
        # Plot the graph
        if test:
            std_data = std_data.query("type != 1")
        else:
            std_data = std_data.query("type != 2")
        sns.boxplot(data=std_data,x="architecture",y="accuracy",hue="type")
        # Get the lines and labels of the graphs
        handles, labels = ax.get_legend_handles_labels()
        # Replace number by real labels
        labels = ["test"*(label == '2.0') + "train"*(label == '0.0') + "validation"*(label == '1.0') for label in labels]
        # Display label information
        ax.legend(handles,labels,fontsize=12)
        xlabels = ax.get_xticklabels()
        xlabels = [self.archi_names[int(float(label.get_text()))] for label in xlabels]
        ax.set_xticklabels(xlabels,fontsize=13)
        ax.set_title(title,fontsize=13)
        ax.set_xticklabels(self.archi_names,fontsize=13)
        ax.set_xlabel("Architectures",fontsize=13)
        ax.set_ylabel("Accuracy",fontsize=13)

#    def plot_evolution(self,archi_name,figure,subplot,test=False,fontsize=13):
#        """
#        Goal:
#        Lineplot - For a given achitecture 
#        Plot the accuracy with respect to the number of epochs
#        Plot both the accuracy on the train set and on the test set
#        Inputs:
#        archi_name = string - name of the architecture 
#                     for which you want to plot the graph
#        figure = matplotlib figure - figure where the boxplot will be plotted
#        subplot = list of size 3 - location of the boxplot in the figure
#        Outputs:
#        """
#        # Set the style
#        sns.set_style("darkgrid")
#        title = archi_name # Title of the graph
#        # Add a subplot in the figure
#        ax = figure.add_subplot(*subplot) 
#        # Get the index assaciated to the name of the architecture 
#        index = self.archi_names.index(archi_name)
#        # Get the data associated to the given architecture
#        archi_data = self.dataframe[self.dataframe["architecture"] == index]
#        # Plot the graph
#        sns.lineplot(data=archi_data,x="epochs",y="accuracy",hue="type",ax=ax,ci=90)
#        handles, labels = ax.get_legend_handles_labels() # Get lines and labels
#        # Replace number by real labels
#        labels = ["test"*(label == '1.0') + "train"*(label == '0.0') for label in labels]
#        # Display label information
#        ax.set_title(title,fontsize=13)
#        ax.set_xlabel("Epochs",fontsize=13)
#        ax.set_ylabel("Accuracy",fontsize=13)
#        ax.legend(handles,labels,fontsize=13)

    def plot_evolution_all(self,figure,subplot,type_perf=0):
        """
        Goal:
        Lineplot - Plot the accuracy with respect to the number of epochs
        Plot the accuracy on the train set or on the test set, not both
        Plot the curves for all architectures
        Inputs:
        figure = matplotlib figure - figure where the boxplot will be plotted
        subplot = list of size 3 - location of the boxplot in the figure
        test = Boolean - True if you want accuracy on test set, False for the train set
        Outputs:
        """
        # Set the style
        sns.set_style("darkgrid")
        # Define the title to be displayed
        subtitle = "test"*(type_perf == 2) + "train"*(type_perf == 0) + "validation"*(type_perf == 1)
        title = "Evolution of the " + subtitle + " accuracy"
        # Create a subplot for the graph
        ax = figure.add_subplot(*subplot)
        # Extracting the right data
        accu_evo = self.dataframe.query("type == " + str(type_perf))
        # Plot the graph
        sns.lineplot(data=accu_evo,x="epochs",y="accuracy",hue="architecture",ax=ax,ci='sd')
        # Get the lines and labels
        handles, labels = ax.get_legend_handles_labels()
        # Replace indexes by real labels
        labels = [self.archi_names[int(float(label))] for label in labels] 
        # Display label information
        ax.legend(handles,labels,fontsize=13)
        ax.set_xlabel("Epochs",fontsize=13)
        ax.set_ylabel("Accuracy",fontsize=13)
        ax.legend(handles,labels,fontsize=13)
        ax.set_title(title,fontsize=13)

    def plot_count_param(self,figure,subplot):
        """
        Goal:
        Plot a barplot of the number of parameters of each model
        Inputs:
        figure = matplotlib figure - figure where the boxplot will be plotted
        subplot = list of size 3 - location of the boxplot in the figure
        Outputs:
        """
        # Set the style
        sns.set_style("darkgrid")
        col1 = "Number of parameters"
        cols = [col1,"Architectures"]
        # Data to be plotted
        data = np.array([self.parameters_count,self.archi_names]).T
        df = pd.DataFrame(data,columns=cols)
        # Convert to numeric
        df[col1] = pd.to_numeric(df[col1])
        ax = figure.add_subplot(*subplot) # Define the ax
        # Plot the graph
        sns.barplot(data=df,x="Architectures",y="Number of parameters",ax=ax)
        labels = ax.get_xticklabels()
        ax.set_xticklabels(labels,fontsize=12)
        ax.set_xlabel("Architectures",fontsize=13)
        ax.set_ylabel(col1,fontsize=13)

    def plot_time_comparison(self,figure,subplot):
        """
        Goal:
        Plot the boxplot of the training time for each model
        Inputs:
        figure = matplotlib figure - figure where the boxplot will be plotted
        subplot = list of size 3 - location of the boxplot in the figure
        """
        # Set the style
        sns.set_style("darkgrid")
        ax = figure.add_subplot(*subplot) # Define the ax
        mean_data_time  = self.datatime.groupby(["architecture"]).mean()
        mean_data_time = mean_data_time.reset_index()
        sns.barplot(data=mean_data_time,x="architecture",y="time",ax=ax)
        # Plot the boxplot
        labels = ax.get_xticklabels()
        labels = [self.archi_names[int(float(label.get_text()))] for label in labels]
        ax.set_xticklabels(labels,fontsize=12)
        ax.set_xlabel("Architectures",fontsize=13)
        ax.set_ylabel("Average training time [s]",fontsize=13)


    def plot_full_comparison(self,test=False,save_folder=None):
        """
        Goal:
        Plot three graphs:
        Evolution of the accuracy on train set for all architectures
        Evolution of the accuracy on test set for all architectures
        Boxplot of the performances with respect to the architecture
        Inputs:
        Outputs:
        """
        type_perf = test*2 + (not test)*1
        # Create the figure
        fig = plt.figure(figsize=[25,14])
        # Plot the evolution on the train set
        self.plot_evolution_all(fig,[3,5,(1,3)],type_perf=0)
        self.plot_count_param(fig,[3,5,(4,5)])
        # Plot the evolution on the test set
        self.plot_evolution_all(fig,[3,5,(6,10)],type_perf=type_perf)
        self.plot_time_comparison(fig,[3,5,(14,15)])
        # Plot the boxplot
        self.plot_std(fig,[3,5,(11,13)],test=test)
        plt.subplots_adjust(wspace=0.5,hspace=0.3)
        if save_folder is not None:
            file_name = "final_plot_" + "validation"*(not test) + test*"test" + ".svg"
            fig.savefig(save_folder + file_name,dpi=250)
        plt.show()

    def plot_errors(self,error_index):
        if (self.errors_target[error_index].item() == 1):
            print('Model predicted right number greater than the left')
        else :
            print('Model predicted left number greater than the right')
        #plot each number with associated prediction
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title('predicted : {0}, True label : {1} ||'.format(self.errors_numbers[error_index,0],self.right_target[error_index,0]))
        axs[0].imshow(self.errors_img[error_index,0,:,:], cmap='gray')
        axs[0].axis('off')
        axs[1].set_title('predicted number : {0}, True label : {1}'.format(self.errors_numbers[error_index,1],self.right_target[error_index,1]))
        axs[1].imshow(self.errors_img[error_index,1,:,:], cmap='gray')
        axs[1].axis('off')
        fig.tight_layout(pad=3.0)
        plt.show()
