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
    target_type = model.target_type
    weights_loss = model.weights_loss
    for epochs in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            if len(target_type) > 1:
                loss_list = []
                for i,target in enumerate(target_type):
                    if target == "target0":
                        partial_loss = weights_loss[i]*criterion(output[i], train_target.narrow(0, b, mini_batch_size))
                    elif target == "target1":
                        partial_loss = weights_loss[i]*criterion(output[i], train_classes.narrow(0, b, mini_batch_size))
                    else:
                        return "Unexpected value in the attribute target_type"
                    loss_list.append(partial_loss)
                loss = sum(loss_list)
            else:
                loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()


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
        with torch.no_grad():
            output = model(input)
            if len(model.target_type) > 1:
                output = output[0]
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
                accuracy_train = self.accuracy(model,train_input,train_target)
                accuracy_test = self.accuracy(model,test_input,test_target)
                row_test = torch.tensor([runs,index,accuracy_test,1,0]).view(1,-1)
                row_train = torch.tensor([runs,index,accuracy_train,0,0]).view(1,-1)
                new_data = torch.cat((new_data,row_train,row_test),dim=0)
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
            row = [archi_name,str(runs),str(round(accuracy_train,1)),str(round(accuracy_test,1))]
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
        header = ["Architecture","Runs","Accuracy Train","Accuracy Test"]
        under_header = ["-"*len(word) for word in header]
        print(self.row_format.format(*header)) # Print the header
        print(self.row_format.format(*under_header)) # Print the header
        for archi_name in self.archi_names:
            self.run_one(archi_name)

    def remove_line(self):
        """
        Goal:
        Inputs:
        Outputs:
        """
        self.dataframe = self.dataframe.query("accuracy < 1e3")

    def reset(self):
        self.dataframe = pd.DataFrame([[1e20,1e20,1e20,1e20,1e20]],columns=self.columns)
    
    def plot_std(self,figure,subplot):
        """
        Goal:
        Inputs:
        Outputs:
        """
        sns.set_style("darkgrid")
        ax = figure.add_subplot(*subplot)
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

    def plot_evolution(self,archi_name,figure,subplot,fontsize=13):
        """
        Goal:
        Inputs:
        Outputs:
        """
        sns.set_style("darkgrid")
        title = archi_name
        ax = figure.add_subplot(*subplot)
        index = self.archi_names.index(archi_name)
        archi_data = self.dataframe[self.dataframe["architecture"] == index]
        sns.lineplot(data=archi_data,x="epochs",y="accuracy",hue="type",ax=ax,ci=90)
        handles, labels = ax.get_legend_handles_labels()
        labels = ["test"*(label == '1.0') + "train"*(label == '0.0') for label in labels]
        ax.set_title(title,fontsize=13)
        ax.set_xlabel("Epochs",fontsize=13)
        ax.set_ylabel("Accuracy",fontsize=13)
        ax.legend(handles,labels,fontsize=13)

    def plot_evolution_all(self,figure,subplot,test=True):
        """
        Goal:
        Inputs:
        Outputs:
        """
        sns.set_style("darkgrid")
        subtitle = "test"*test + "train"*(not test)
        title = "Evolution of the " + subtitle + " accuracy"
        ax = figure.add_subplot(*subplot)
        if test:
            accu_evo = self.dataframe.query("type == 1")
        else:
            accu_evo = self.dataframe.query("type == 0")
        sns.lineplot(data=accu_evo,x="epochs",y="accuracy",hue="architecture",ax=ax,ci=90)
        handles, labels = ax.get_legend_handles_labels()
        labels = [self.archi_names[int(float(label))] for label in labels] 
        ax.legend(handles,labels,fontsize=13)
        ax.set_xlabel("Epochs",fontsize=13)
        ax.set_ylabel("Accuracy",fontsize=13)
        ax.legend(handles,labels,fontsize=13)
        ax.set_title(title,fontsize=13)

    def plot_full_comparison(self):
        """
        Goal:
        Inputs:
        Outputs:
        """
        fig = plt.figure(figsize=[16,10])
        self.plot_evolution_all(fig,[2,2,1],test=False)
        self.plot_evolution_all(fig,[2,2,2])
        self.plot_std(fig,[2,2,(3,4)])
        plt.show()
        

        
        

