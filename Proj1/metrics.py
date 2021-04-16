import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
from dlc_practical_prologue import generate_pair_sets

def train_model(model, train_input, train_target, train_classes,
                nb_epochs=50, 
                mini_batch_size = 100, 
                eta = 0.1, 
                criterion = nn.CrossEntropyLoss()):
    optimizer = torch.optim.Adam(model.parameters(), lr = eta)
    for epochs in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
     
"""
def compute_nb_errors(model, data_input, data_target, mini_batch_size=100):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors



def plot_error_vs_epochs(model, max_nb_epochs, step, train_input, train_target, test_input, test_target):

    To be checked.
    Goal: plot number of errors in function of nb epochs on the stream ie while training the model (for efficiency reasons)

    error_rates = []
    
    for nb_epochs in range(0, max_nb_epochs//step):
        # Train model for 'step' epochs, compute error at each iteration
        train_model(model, train_input, train_target, nb_epochs=step)
        error = compute_nb_errors(model, test_input, test_target)/test_input.size(0)
        error_rates.append(error)
        
    plt.plot(np.arange(0, max_nb_epochs, step), error_rates)
    plt.xlabel("number of epochs")
    plt.ylabel("error rate")
    plt.title("Naive Net")
    plt.show()
"""

class Cross_validation():

    def __init__(self,
                 architectures,
                 args,
                 steps=None,
                 runs=10,load=5000,epochs=50):
        """
        Goal:
        Inputs:
        architectures = list of class generating the architectures
        args = list of the arguments of each class
        steps =
        runs =
        load =
        epochs =
        Outputs:
        """
        self.architectures = architectures
        self.archi_names = [archi.__name__ for archi in self.architectures]
        self.args = args
        self.runs = runs
        self.columns = ["run_id","architecture","accuracy","type","epochs"]
        self.dataframe = pd.DataFrame([[1e20,1e20,1e20,1e20,1e20]],columns=self.columns)
        data = generate_pair_sets(load)
        self.size = 1000
        self.epochs = epochs
        self.train_input, self.train_target, self.train_classes = data[0], data[1], data[2]
        self.test_input, self.test_target, self.test_classes = data[3], data[4], data[5]
        self.steps = steps
        self.row_format = '{:<20}{:<15}{:<25}{:<25}' # Define the display format

    def split_data(self):
        """
        Goal:
        Extract 1000 (size attribute) training and testing data points
        Inputs:
        Outputs:
        train_input = 
        train_target = 
        train_classes = 
        test_input = 
        test_target = 
        test_classes = 
        """
        shuffle = torch.randperm(self.train_input.shape[0])
        index = shuffle[:self.size]
        train_input = self.train_input[index]
        train_target = self.train_target[index]
        train_classes = self.train_classes[index]
        test_input = self.test_input[index]
        test_target = self.test_target[index]
        test_classes = self.test_classes[index]
        return train_input, train_target, train_classes ,test_input ,test_target ,test_classes

    def accuracy(self,model,input,target):
        """
        Goal:
        Inputs:
        Outputs:
        """
        output = model(input)
        _,predicted = torch.max(output,dim=1)
        errors = torch.where(target != predicted,1,0).sum().item()
        accuracy = (1 - errors/(target.shape[0]))*100
        return accuracy

    def run_one(self,archi_name):
        """
        Goal:
        Inputs:
        Outputs:
        """
        if not archi_name in self.archi_names:
            return "Unexpected value for archi_name"
        index = self.archi_names.index(archi_name)
        Myclass = self.architectures[index]
        args = self.args[index]
        for runs in range(self.runs):
            data = self.split_data()
            train_input, train_target, train_classes = data[0], data[1], data[2]
            test_input, test_target, _ = data[3], data[4], data[5]
            model = Myclass(*args)
            new_data = torch.zeros(len(self.columns)).view(1,-1)
            if self.steps is not None:
                for step in range(self.steps,self.epochs,self.steps):
                    train_model(model, 
                                train_input, 
                                train_target, 
                                train_classes, 
                                nb_epochs=self.steps)
                    accuracy_train = self.accuracy(model,train_input,train_target)
                    accuracy_test = self.accuracy(model,test_input,test_target)
                    row_test = torch.tensor([runs,index,accuracy_test,1,step]).view(1,-1)
                    row_train = torch.tensor([runs,index,accuracy_train,0,step]).view(1,-1)
                    new_data = torch.cat((new_data,row_train,row_test),dim=0)
            else:
                train_model(model, 
                            train_input, 
                            train_target, 
                            train_classes, 
                            nb_epochs=self.epochs)
                accuracy_train = self.accuracy(model,train_input,train_target)
                accuracy_test = self.accuracy(model,test_input,test_target)
                row_test = torch.tensor([runs,index,accuracy_test,1,self.epochs]).view(1,-1)
                row_train = torch.tensor([runs,index,accuracy_train,0,self.epochs]).view(1,-1)
                new_data = torch.cat((new_data,row_train,row_test),dim=0)
            run_str = "runs n°" + str(runs)
            accu_str_train = "accuracy train = " + str(round(accuracy_train,1))
            accu_str_test = "accuracy test = " + str(round(accuracy_test,1))
            row = [archi_name,run_str,accu_str_train,accu_str_test]
            print(self.row_format.format(*row)) # Print the header
            new_data = new_data[1:]
            df = pd.DataFrame(data=new_data.tolist(),columns=self.columns)
            self.dataframe = self.dataframe.append(df,ignore_index=True)
        self.remove_line()

    def run_all(self):
        """
        Goal:
        Inputs:
        Outputs:
        """
        for archi_name in self.archi_names:
            self.run_one(archi_name)

    def remove_line(self):
        """
        Goal:
        Inputs:
        Outputs:
        """
        self.dataframe = self.dataframe.query("accuracy < 1e3")
    
    def plot_std(self,figure,subplot):
        """
        Goal:
        Inputs:
        Outputs:
        """
        sns.set_style("darkgrid")
        ax = figure.add_subplot(subplot)
        title = "Results (epochs = " + str(self.epochs) + ")"
        max_epochs = self.dataframe["epochs"].max()
        std_data = self.dataframe.query("epochs == " + str(max_epochs))
        sns.boxplot(data=std_data,x="architecture",y="accuracy",hue="type")
        handles, labels = ax.get_legend_handles_labels()
        labels = ["test"*(label == '1.0') + "train"*(label == '0.0') for label in labels]
        ax.legend(handles,labels,fontsize=13)
        ax.set_title(title,fontsize=13)
        ax.set_xticklabels(self.archi_names,fontsize=13)
        ax.set_xlabel("Architectures",fontsize=13)
        ax.set_ylabel("Accuracy",fontsize=13)
        plt.show()

    def plot_evolution(self,archi_name,figure,subplot,fontsize=13):
        """
        Goal:
        Inputs:
        Outputs:
        """
        sns.set_style("darkgrid")
        title = archi_name
        ax = figure.add_subplot(subplot)
        index = self.archi_names.index(archi_name)
        archi_data = self.dataframe[self.dataframe["architecture"] == index]
        sns.lineplot(data=archi_data,x="epochs",y="accuracy",hue="type",ax=ax,ci=90)
        handles, labels = ax.get_legend_handles_labels()
        labels = ["test"*(label == '1.0') + "train"*(label == '0.0') for label in labels]
        ax.set_title(title,fontsize=13)
        ax.set_xlabel("Epochs",fontsize=13)
        ax.set_ylabel("Accuracy",fontsize=13)
        ax.legend(handles,labels,fontsize=13)
        plt.show()
        

        
        

