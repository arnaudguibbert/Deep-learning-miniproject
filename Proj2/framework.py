from torch import empty
import math

#################### Linear class #################### 
class Linear():
    
    def __init__(self,in_size,out_size,add_bias=True):
        """
        Goal:
        Inputs:
        in_size = int > 0 - Input size of the layer
        out_size = int > 0 - Output size of the layer
        add_bias = bool - Specify if you want to add a bias or not
        Outputs:
        """
        # Initialize the weights and the associated gradient with Xavier Initialisation
        xavier = math.sqrt(6/(in_size+out_size))
        self.weights = empty(out_size,in_size).uniform_(-xavier,xavier)
        # Initialize the weights to 0
        self.grdweights = empty(out_size,in_size).fill_(0)
        if add_bias: 
            # Initialize the bias at 0
            self.bias = empty(out_size,1).fill_(0) 
            self.grdbias = empty(out_size,1).fill_(0)
        else:
            # Else set it to None
            self.bias = None
            self.grdbias = None
        # Initialize the inputs attribute
        self.inputs = []
        # Initialize the back flag (true if backpropagation has been performed)
        self.back = False
            
    def forward(self,inputs,no_grad=False):
        """
        Goal:
        Perform the forward step
        Inputs:
        inputs = torch tensor - size NxDin (N number of datapoints, Din input size of the layer)
        Outputs:
        output = torch tensor - size NxDout (N number of datapoints, Dout output size of the layer)
        """
        # Store the input into the input attribute (will be useful for the backward step)
        if not no_grad:
            self.inputs.append(inputs) 
        # Compute the output
        output = inputs@(self.weights.T)
        if self.bias is not None:
            output += self.bias.T # Add bias
        return output
        
    def backward(self,grdwrtoutput):
        """
        Goal:
        Perform the backward step (does not change the weights to do so look at optimization_step)
        Inputs:
        grdwrtoutput = torch tensor - size NxDout (N number of datapoints, Dout input size of the layer)
        Outputs:
        grdwrtinput = torch tensor - size NxDin (N number of datapoints, Din input size of the layer)
        """
        # if the forward step has not been performed raise an error message
        if len(self.inputs) == 0:
            return "Forward step has not been performed"
        inputs = self.inputs.pop()
        # Compute the gradient with respect to the input and accumulate
        grdwrtinput = grdwrtoutput@self.weights 
        # Compute the gradient with respect to the weights
        self.grdweights += (grdwrtoutput.T)@inputs
        if self.bias is not None:
            # Compute the gradient with respect to the bias and accumulate
            self.grdbias += grdwrtoutput.sum(dim=0).view(-1,1)
        # Set the flag back to true --> means ready for the optimization step
        self.back = True
        return grdwrtinput
    
    def zero_grad(self):
        """
        Goal:
        Reset the gradient tensors to zero
        Inputs:
        Outputs:
        """
        # Reset the gradient tensors
        self.grdweights = empty(self.weights.shape).fill_(0)
        if self.bias is not None:           
            self.grdbias = empty(self.bias.shape).fill_(0)
        
    def optimization_step(self,lr):
        """
        Goal:
        Perform one optimization_step
        Inputs:
        lr = float > 0 - learning rate 
        Outputs:
        """
        # If the backward step has not been performed raise an error message
        if not self.back:
            return "Backward step has not been performed"
        # Update the weights
        self.weights -= self.grdweights*lr
        if self.bias is not None:
            # Update the bias
            self.bias -= self.grdbias*lr
        # Set the flag back to 0
        self.back = False
        
    def reset(self):
        """
        Goal:
        Reset the weigths and bias, reset also the gradient tensors to 0, reset the inputs attribute
        Inputs:
        Outputs:
        """
        in_size, out_size = self.weights.shape # Get the shape of the layer
        xavier = math.sqrt(6/(in_size+out_size)) # Xavier Initialization
        # Reset the weights
        self.weights = empty(self.weights.shape).uniform_(-xavier,xavier)
        if self.bias is not None:
            # Reset the bias
            self.bias = empty(self.bias.shape).fill_(0)
        # Set gradients to 0
        self.zero_grad()
        # Reset the inputs to None
        self.inputs = None
        
    @property
    def params(self):
        """
        Goal:
        Return the weights, bias and their respective gradient tensor
        Inputs:
        Outputs:
        param = torch tensor - 
                size DoutxDin [+1 if bias] (Dout output size of the layer, Din input size of the layer)
        grdparam = torch tensor - 
                   size DoutxDin [+1 if bias] (Dout output size of the layer, Din input size of the layer)
        """
        if self.bias is None:
            param = self.weights # Get the weights
            grdparam = self.grdweights # Get the gradient
        else:
            param = [self.weights,self.bias] # To be modified
            grdparam = [self.grdweights,self.grdbias] # To be modified
        return param, grdparam

#################### MSELoss class #################### 

class MSELoss():
    
    def __init__(self,mean=True):
        """
        Goal:
        Inputs:
        mean = bool - specify if the loss is a mean or a sum
        Outputs:
        """
        self.mean = mean
        # Initialize the inputs attribute
        self.inputs = None
        # Intialize the targets attribute
        self.targets = None
       
    def forward(self,inputs,targets):
        """
        Goal:
        Compute the loss
        Inputs:
        inputs = torch tensor - (N number of datapoints, D output size of the last layer)
        targets = torch tensor - (N number of datapoints, D output size of the last layer)
        Outputs:
        loss = float - MSE loss 
        """
        # Store the inputs into the inputs attributes (will be useful for the backward)
        self.inputs = inputs
        # Store the targets into the inputs attributes (will be useful for the backward)
        self.targets = targets
        if self.mean:
            # Compute the loss and take the mean
            loss = ((inputs - targets)**2).mean()
        else:
            # Compute the loss and take the sum
            loss = ((inputs - targets)**2).sum()
        return loss
            
    def backward(self):
        """
        Goal:
        Compute the gradient of the loss with respect to the output of the last layer
        Inputs:
        Outputs:
        """
        # Raise a message error if the forward step has not been performed
        if self.inputs is None or self.targets is None:
            return "Forward step has not been performed"
        if self.mean:
            # Compute the gradient 
            grdwrtinput = 2*(self.inputs-self.targets)/(self.inputs.shape[0])
        else:
            grdwrtinput = 2*(self.inputs-self.targets)
        return grdwrtinput    
      
    @property
    def params(self):
        """
        Goal:
        Return the parameters, actually nothing to return
        Inputs:
        Outputs:
        """
        return []

#################### ReLU class #################### 

class ReLU():
    
    def __init__(self):
        """
        Goal:
        Inputs:
        Outputs: 
        """
        # Initialize the list of inputs attribute
        self.inputs = []
        
    def forward(self,inputs,no_grad=False):
        """
        Goal:
        Perform the forward step using ReLU
        Inputs:
        inputs = torch tensor
        Outputs: 
        torch tensor of the same size
        """
        if not no_grad:
            # Add inputs to the attribute "input"
            self.inputs.append(inputs)
        # Set all negative components of input to zero to get ReLU of the inputs
        output = inputs.clone()
        output[inputs <= 0] = 0
        return output
        
    def backward(self,grdwrtoutput):
        """
        Goal: 
        Perform the backward step after ReLU has been computed
        Inputs: 
        gradient with respect to output - torch tensor of size NxD where D is the dimension of the layer and N the number of samples
        Outputs: 
        torch tensor of the same size storing the gradient with respect to the input
        """
        if len(self.inputs) == 0:
            return "Forward step has not been performed"
        inputs = self.inputs.pop()
        # grdphi contains the component-wise gradient of ReLU
        grdphi = empty(inputs.shape).fill_(0)
        grdphi[inputs > 0] = 1
        # Use chain-rule to compute gradient w.r.t. input from gradient w.r.t. output
        grdwrtinput = grdphi*grdwrtoutput
        return grdwrtinput
        
    @property
    def params(self):
        """
        Goal:
        Return the parameters, actually nothing to return
        Inputs:
        Outputs:
        """
        return []

#################### Tanh class #################### 

class Tanh():
    
    def __init__(self):
        """
        Goal:
        Inputs:
        Outputs: 
        """
        # Initialize the inputs attribute
        self.inputs = []
        
    def tanh(self,x):
        """
        Goal:
        Inputs: 
        x = torch tensor
        Outputs: 
        Component-wise tanh of the input = torch tensor of the same size
        """
        return (x.exp() - (-x).exp())/(x.exp() + (-x).exp())
        
    def forward(self,inputs,no_grad=False):
        """
        Goal: 
        Perform the forward step computing tanh
        Inputs: 
        inputs = torch tensor
        Outputs: 
        Torch tensor of the same size = tanh(input)
        """
        if not no_grad:
            # Add inputs to the attribute "input"
            self.inputs.append(inputs)
        # Compute tanh of the input
        output = self.tanh(inputs)
        return output
        
    def backward(self,grdwrtoutput):
        """
        Goal: 
        Perform the backward step after tanh has been computed
        Inputs: 
        gradient with respect to output - torch tensor of size NxD where D is the dimension of the layer and N the number of samples
        Outputs: 
        torch tensor of the same size storing the gradient with respect to the input
        """
        if len(self.inputs) == 0:
            return "Forward step has not been performed"
        inputs = self.inputs.pop()
        # Compute the gradient using chain rule and return it
        grdwrtinput = (1 - self.tanh(inputs)**2)*grdwrtoutput
        return grdwrtinput
        
    @property
    def params(self):
        """
        Goal:
        Return the parameters, actually nothing to return
        Inputs:
        Outputs:
        """
        return []

#################### Sequential class #################### 

class Sequential():
    
    def __init__(self,*sequence):  # deleted the * before "sequence" so that a list of modules works
        """
        Goal:
        Inputs: 
        sequence = list of modules
        loss = module of a loss
        Outputs:
        """
        # Initialize the sequence attribute, containing modules of models and losses
        self.sequence = sequence
        # Initialize the back flag (true if backpropagation has been performed)
        self.back = False
        
    def forward(self,inputs,no_grad=False):
        """
        Goal:
        Perform the forward path
        Inputs:
        inputs = list of modules
        Outputs:
        output of the forward path = torch tensor
        """
        # Initialize the variable that will later contain the output
        output = inputs.clone()
        # Perform the forward path by calling the forward method of every module in the sequence
        for module in self.sequence:
            output = module.forward(output,no_grad=no_grad)
        return output
        
    def backward(self,grdwrtoutput):
        """
        Goal:
        Perform the backward path by updating the gradients
        Inputs:
        grdwrtoutput = torch tensor of size number of samples x size of the last layer // gradient with respect to the output
        Outputs:
        """
        for module in reversed(self.sequence):
            # Compute the gradients
            grdwrtoutput = module.backward(grdwrtoutput)
            if grdwrtoutput == "Forward step has not been performed":
                message = str(type(module).__name__) + " : " + grdwrtoutput
                return message
        # Set the backward flag to true
        self.back = True
            
    def optimization_step(self,lr):
        """
        Goal: 
        update the weights using module.optimization_step for all the linear modules involved
        Inputs: 
        lr = float > 0: learning rate
        Outputs:
        """
        # If the backward step hasn't been performed, return an error message
        if not self.back:
            return "Backward step has not been performed"
        # For every module that can be optimized, i.e. for the instances of Linear, update the weights
        for module in self.sequence:
            if hasattr(module,"optimization_step"):
                module.optimization_step(lr)      
        self.back = False
                
    def zero_grad(self):
        """
        Goal: 
        set all gradients to zero
        Inputs: 
        Outputs:
        """
        for module in self.sequence:
            # Use the method zero_grad of all involved modules, which fulfill that purpose
            if hasattr(module,"zero_grad"):
                module.zero_grad()

    def reset(self):
        """
        Goal: 
        reset all parameters to their default value, reset the gradients to 0, reset the inputs attribute
        Inputs: 
        Outputs:
        """
        for module in self.sequence:
            # If the module has the attribute "reset", namely if its class is Linear, apply its reset method
            if hasattr(module,"reset"):
                module.reset()
    
    @property
    def params(self):
        """
        Goal:
        Return all parameters, i.e. weights and biases of each layer, and gradients of the weights and biases
        Inputs:
        Outputs:
        """
        # Initialize the list of all paremeters
        params = []
        for module in self.sequence:
            mod_params = module.params
            # For every module, if the current module has parameters, add them
            if len(mod_params) > 0:
                params.append(mod_params)
        return params
       