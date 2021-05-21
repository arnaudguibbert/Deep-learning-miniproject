from pandas.core.algorithms import unique
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
try:
    import pandas as pd
except ModuleNotFoundError:
    pass
try:
    import seaborn as sns
except:
    pass
import numpy as np
from time import perf_counter
from dlc_practical_prologue import generate_pair_sets

def normalize(data,mean=None,std=None):
    """
    Goal:
    Normalize the data - idest substracting the mean, divide by the standard deviation
    Inputs:
    data = torch tensor - size Nx2xHxW (N number of data points, H height of the image, W width of the image)
           the first column of the data shall be the classes
    mean = torch tensor - size 1x2xHxW (H height of the image, W width of the image)
           if the mean tensor is passed, then it will directly use this mean tensor instead of computing it
    std = torch tensor - size 1x2xHxW (H height of the image, W width of the image)
          if the std tensor is passed, then it will directly use this std tensor instead of computing it
    Outputs:
    data = torch tensor - size Nx2xHxW (N number of data points, H height of the image, W width of the image)
           Normalized data
    mean = torch tensor - size 1x2xHxW (H height of the image, W width of the image)
           Mean of the data, or mean passed as argument
    std = torch tensor - size 1x2xHxW (H height of the image, W width of the image)
          standard deviation of the data, or std passed as argument
    """
    
    if mean is None:
        mean = torch.mean(data,dim=0,keepdim=True) # Compute the mean
    if std is None: 
        std = torch.std(data,dim=0,keepdim=True) # Compute the std
    #std=torch.where(std==0.0,0.1,std  )
    norm_data = data.clone() 
    std = torch.where(std==0,1.,std.double()).float()
    norm_data = (norm_data - mean)/std # Normalize
    return norm_data, mean, std

def train_model(model, train_input, train_target, train_classes,
                nb_epochs=50, 
                batch_size = 100, 
                eta = 0.05):
    """
    Goal:
    Train a given model 
    Inputs:
    model = nn.Module class object - model to be trained
    train_input = tensor - size (Nx2x14x14) (N number of samples)
                  input of the training datas
    train_target = tensor - size (N) (N number of samples)
                   targets - belongs to {0,1}
    train_classes = tensor - size (Nx2) (N number of samples)
                    Classes (i.e. numbers) of the two images - belongs to {1,...,10}
    nb_epochs = int - Number of epochs for the training
    batch_size = int - size of the batches
    eta = float - learning rate 
    Outputs:
    """
    model.train()
    criterion = nn.CrossEntropyLoss() # Define the loss
    optimizer = torch.optim.Adam(model.parameters(), lr = eta) # Define the optimizer
    # For each output of the model a target type is associated (auxiliary losses)
    target_type = model.target_type 
    # For each loss a weight is defined
    weights_loss = model.weights_loss
    for epochs in range(nb_epochs):
        for b in range(0, train_input.size(0), batch_size):
            # Compute the output of the model
            output = model(train_input.narrow(0, b, batch_size))
            # If there is multiple outputs (auxiliary losses)
            if len(target_type) > 1:
                loss_list = [] # List where the losses will be stored
                for i,target in enumerate(target_type):
                    # Target 0 means that the output predicts whether the first number 
                    # is higher or lower than the second one
                    if target == "target0":
                        # Compute the auxiliary loss
                        aux_loss = criterion(output[i], 
                                             train_target.narrow(0, b, batch_size))
                    # Target 1 means that the output predicts the classes of the images
                    elif target == "target1":
                        # Compute the auxiliary loss
                        aux_loss = criterion(output[i], 
                                             train_classes.narrow(0, b, batch_size))
                    else:
                        # print Error message
                        return "Unexpected value in the attribute target_type"
                    # Add the weighted auxiliary loss to the loss list
                    loss_list.append(aux_loss*weights_loss[i])
                # Sum the auxiliary losses
                loss = sum(loss_list)
            else:
                # Else if one output : no auxiliary losses
                loss = criterion(output, train_target.narrow(0, b, batch_size))
            model.zero_grad() # Reset the gradient tensors to 0
            loss.backward() # Perform a backward step
            optimizer.step() # Update the weights

def std_accuracy(data_path,save_data=None):
    """
    Goal:
    Inputs:
    Outputs:
    """
    columns = ["Architecture index","Mean","Std"]
    row_format = '{:<20}{:<15}{:<15}'
    data = np.genfromtxt(data_path,delimiter=",",skip_header=1).astype(float)
    max_epochs = np.max(data[:,-1])
    data = data[data[:,-1] == max_epochs]
    unique_archi = np.unique(data[:,1])
    new_data = np.zeros((unique_archi.shape[0],3))
    new_data[:,0] = unique_archi
    print(row_format.format(*columns))
    "data_architectures/metrics" + save_data + ".csv"
    for i,archi in enumerate(unique_archi):
        data_archi = data[data[:,1] == archi]
        new_data[i,1] = np.mean(data_archi[:,2])
        new_data[i,2] = np.std(data_archi[:,2])
        row_display = [int(archi),
                       round(new_data[i,1],2),
                       round(new_data[i,2],2)]
        print(row_format.format(*row_display))
    if save_data is not None:
        name_file = "data_architectures/metrics" + save_data + ".csv"
        np.savetxt(name_file,new_data,delimiter=",",header = ",".join(columns))


class Cross_validation():

    def __init__(self,
                 architectures,
                 args,
                 steps=5,
                 runs=10,load=5000,epochs=50,pandas_flag=False):
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
        if pandas_flag:
            self.datatime = pd.DataFrame([[1e20,1e20,1e20]],columns=self.columns_time)
            self.dataframe = pd.DataFrame([[1e20,1e20,1e20,1e20,1e20]],columns=self.columns)
        else:
            self.datatime = np.zeros((0,len(self.columns_time)))
            self.dataframe = np.zeros((0,len(self.columns)))
        # Load the the data set
        data = generate_pair_sets(load)
        self.size = 1000 # Number of samples used for training and testing at each run
        self.epochs = epochs # get the number of epochs
        # To be checked (are these two lines necessary ??)
        self.train_input, self.train_target, self.train_classes = data[0], data[1], data[2]
        self.test_input, self.test_target, self.test_classes = data[3], data[4], data[5]
        self.steps = steps # Get Granularity for the graphs
        # Row format for the logs
        self.row_format = '{:<20}{:<15}{:<25}{:<25}{:<15}' # Define the display format
        self.data_count = None
        self.pandas_flag = pandas_flag
        #store it to see plot where the model fail, random initialisation
        self.errors_img = torch.empty(0,1) # torch.empty() marche aussi ?
        self.errors_target = torch.empty(0,1)
        self.errors_numbers = torch.empty(0,1)
        self.right_target = torch.empty(0,1)

    def count_params(self,save_data=None):
        """
        Goal:
        Count the number of parameters to be trained for each model. Save these datas
        into the data_architectures folder 
        Inputs:
        save_data = string - name of the file 
        Outputs:
        """
        param_count = []
        for i,archi in enumerate(self.architectures):
            model = archi(*self.args[i])
            n_params = 0
            with torch.no_grad():
                for params in model.parameters():
                    n_params += params.numel()
            param_count.append(n_params)
        data_param = torch.tensor(param_count).view(-1,1)
        index = torch.arange(len(self.architectures)).view(-1,1)
        data_param = torch.cat((index,data_param),dim=1)
        self.data_count = data_param
        if save_data is not None:
            name_file = "data_architectures/param_count" + save_data + ".csv"
            columns=["Architectures index","Number of parameters"]
            if self.pandas_flag:
                param_pd = pd.DataFrame(data_param.tolist(),
                                        columns=columns)
                param_pd.to_csv(name_file,index=False)
            else:
                data_np = data_param.numpy()
                print(data_np)
                np.savetxt(name_file,data_np,delimiter=",",header=",".join(columns))

    def index_for_equal_class(self,targets):
        """
        Goal:
        Determine the indexes to choose such that the classes are balanced 
        in the data set. A random shuffle is first applied
        Inputs:
        targets = torch tensor - size N (number of data points)
                                 label of the data points 
        Outputs:
        indexes = torch tensor - size self.size 
                                 indexes selected
        """
        index_class0 = (targets == 0).nonzero(as_tuple=False).view(-1) # Select the classes
        index_class1 = (targets == 1).nonzero(as_tuple=False).view(-1)
        index_class0 = index_class0[torch.randperm(index_class0.shape[0])] # Shuffle the indexes
        index_class1 = index_class1[torch.randperm(index_class1.shape[0])] # Shuffle the indexes
        indexes = torch.cat((index_class0[:self.size//2],index_class1[:self.size//2]))
        indexes = indexes[torch.randperm(indexes.shape[0])] # Shuffle the final indexes
        return indexes


    def split_data(self,test=False):
        """
        Split the data into a train/(validation or test) of size self.size (each set)
        Preserve the equal class distribution (50% class 0, 50% class 1)
        Normalize the input datas
        Goal:
        Extract randomly 1000 (size attribute) training and testing/validation data points
        Inputs:
        test = specify if you want test data set or validation data set for the second element
        Outputs:
        train_input = tensor - size (1000x2x14x14)
                      input of the training datas
        train_target = tensor - size (1000)
                       targets - belongs to {0,1}
        train_classes = tensor - size (1000x2)
                        Classes (i.e. numbers) of the two images - belongs to {1,...,10}
        test_input = tensor - size (1000x2x14x14)
                      input of the validation datas
        test_target = tensor - size (1000)
                       targets - belongs to {0,1}
        test_classes = tensor - size (1000x2)
                        Classes (i.e. numbers) of the two images - belongs to {1,...,10}
        """
        shuffle_index = torch.randperm(self.train_target.shape[0])
        load = shuffle_index.shape[0]
        train_input_shuffle = self.train_input[shuffle_index]
        train_target_shuffle = self.train_target[shuffle_index]
        train_classes_shuffle = self.train_classes[shuffle_index]
        index_train = self.index_for_equal_class(train_target_shuffle[:load//2])
        train_input = train_input_shuffle[index_train]
        train_target = train_target_shuffle[index_train]
        train_classes = train_classes_shuffle[index_train]
        if not test:
            index_test = self.index_for_equal_class( train_target_shuffle[load//2:]) + load//2
            test_input = train_input_shuffle[index_test]
            test_target = train_target_shuffle[index_test]
            test_classes = train_classes_shuffle[index_test]
        else:
            index_test = self.index_for_equal_class(self.test_target)
            test_input = self.test_input[index_test]
            test_target = self.test_target[index_test]
            test_classes = self.test_classes[index_test]
        train_input, mean, std = normalize(train_input)
        test_input, _, _ = normalize(test_input,mean,std)
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
            _,predicted = torch.max(output,dim=1) # Compute the prediction
            #compute the error matrix
            errors_matrix = torch.where(target != predicted,1,0)
            # Compute the number of errors
            total_errors = errors_matrix.sum().item()
            """
            #store the wrong set of image
            errors_index = torch.empty(0,1)
            errors_index = ((errors_matrix == 1).nonzero(as_tuple=True)[0])
            self.errors_img = input[errors_index]
            self.errors_target = predicted[errors_index]
            if len(model.target_type) > 1 and len(errors_index) != 0: 
                #We can see the errors only if the model has it as output
                self.errors_numbers=torch.argmax(output[1][errors_index],dim=1)
                self.right_target=target_classes[errors_index]
            """
            # Compute the accuracy
            accuracy = (1 - total_errors/(target.shape[0]))*100
        model.train()
        return accuracy

    """
    def get_errors(self):
        return self.errors_img, self.errors_target, self.errors_numbers
    """

    def run_one(self,archi_name,test=False,save_weight=None):
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
        #save the model weight
        if save_weight is not None:
            torch.save(model.state_dict(), 'model/{}_weights.pth'.format(archi_name))
        # Remove the first artificial line
        new_data = new_data[1:]
        new_data_time = new_data_time[1:]
        # Add the new data to the existing data frame
        if self.pandas_flag:
            df = pd.DataFrame(data=new_data.tolist(),columns=self.columns)
            self.dataframe = self.dataframe.append(df,ignore_index=True)
            # Remove the first artificial line of the data frame
            df_time = pd.DataFrame(data=new_data_time.tolist(),columns=self.columns_time)
            self.datatime = self.datatime.append(df_time,ignore_index=True)
        else:
            self.datatime = np.concatenate((self.datatime,new_data_time.numpy()),axis=0)
            self.dataframe = np.concatenate((self.dataframe,new_data.numpy()),axis=0)
        self.remove_line()

    def run_all(self,test=False,save_data=None):
        """
        Goal:
        For each architecture : 
        Train a model generated by the architecture "self.runs" times.
        Store the performances of these models in the data frame
        the performances are recorded each "self.step" epochs
        Inputs:
        test = Boolean - are you validating hyperparameters or perform a final test 
               on the testing data
        save_data = string - file name for the data to be stored
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
            self.run_one(archi_name,test=test,save_weight=save_data)
        if save_data is not None:
            name_file_corres = "data_architectures/corres_index" + save_data + ".csv"
            name_file_accuracy = "data_architectures/accuracy" + save_data + ".csv"
            name_file_time = "data_architectures/time" + save_data + ".csv"
            if self.pandas_flag:
                corres_pd = pd.DataFrame(self.archi_names,columns=["Architecture name"])
                corres_pd.to_csv(name_file_corres)
                self.dataframe.to_csv(name_file_accuracy,index=False)
                self.datatime.to_csv(name_file_time,index=False)
            else:
                np.savetxt(name_file_accuracy,
                           self.dataframe,delimiter=",",
                           header=",".join(self.columns))
                np.savetxt(name_file_time,
                           self.datatime,delimiter=",",
                           header=",".join(self.columns_time))

    def remove_line(self):
        """
        Goal:
        Remove the artficial first line of the data frame
        Inputs:
        Outputs:
        """
        if self.pandas_flag:
            # Remove the first line where accuracy was set to 1e20
            self.dataframe = self.dataframe.query("accuracy < 1e3")
            self.datatime = self.datatime.query("time < 1e10")

    def reset(self):
        """
        Goal:
        Reset the data acquired so far (warning it will erase the content)
        Inputs:
        Outputs:
        """
        # Reset the data frame
        if self.pandas_flag:
            self.dataframe = pd.DataFrame([[1e20,1e20,1e20,1e20,1e20]],columns=self.columns)
            self.datatime = pd.DataFrame([[1e20,1e20,1e20]],columns=self.columns_time)
        else:
            self.dataframe = np.zeros((0,self.dataframe.shape[1]))
            self.datatime = np.zeros((0,self.datatime.shape[1]))
    
    def plot_std(self,figure,subplot,test=False):
        """
        Goal:
        Boxplot - Plot the standard deviations of the performances of each
        architectures on the training and testing set after having been trained
        for self.epochs epochs
        Inputs:
        figure = matplotlib figure - figure where the boxplot will be plotted
        subplot = list of size 3 - location of the boxplot in the figure
        test = Boolean - are you validating hyperparameters or perform a final test 
               on the testing data
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

    def plot_evolution_all(self,figure,subplot,type_perf=0):
        """
        Goal:
        Lineplot - Plot the accuracy with respect to the number of epochs
        Plot the accuracy on the train set or on the test set, not both
        Plot these curves for all architectures
        Inputs:
        figure = matplotlib figure - figure where the boxplot will be plotted
        subplot = list of size 3 - location of the boxplot in the figure
        type_perf = int included in {0,1,2} - 0 if you want to plot the train curves
                                              1 if you want to plot the validation curves
                                              2 if you want to plot the test curves
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
        test = Boolean - are you validating hyperparameters or perform a final test 
               on the testing data
        Outputs:
        """
        type_perf = test*2 + (not test)*1
        # Create the figure
        fig = plt.figure(figsize=[25,14])
        # Plot the evolution on the train set
        self.plot_evolution_all(fig,[2,6,(1,3)],type_perf=0)
        # Plot the evolution on the test set
        self.plot_evolution_all(fig,[2,6,(4,6)],type_perf=type_perf)
        self.plot_time_comparison(fig,[2,6,(11,12)])
        # Plot the boxplot
        self.plot_std(fig,[2,6,(7,10)],test=test)
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
