from torch import empty
import math

class FrameworkModule():

    def __init__(self):
        self.inputs = []
        pass

    def forward(self,inputs,no_grad=False):
        """
        Goal:
        Perform the forward step
        Inputs:
        inputs = torch tensor - size NxDin (N number of datapoints, Din input size of the layer)
        Outputs:
        output = torch tensor - size NxDout (N number of datapoints, Dout output size of the layer)
        """
        pass

    def backward(self,grdwrtoutput):
        """
        Goal:
        Perform the backward step (does not change the weights to do so look at optimization_step)
        Inputs:
        grdwrtoutput = torch tensor - size NxDout (N number of datapoints, Dout input size of the layer)
        Outputs:
        grdwrtinput = torch tensor - size NxDin (N number of datapoints, Din input size of the layer)
        """
        pass 

    def update(self,new_params):
        """
        Goal:
        Replace the parameters of the model by the new parameters
        Inputs:
        new_params = dict - dictionary having, at least, as key the module itself. The value associated
                     to this key is a list containing the new parameters tensor. If the model is parameter-less
                     then the key value is None
        Outputs:
        """
        pass

    def reset(self):
        """
        Goal:
        Reset the parameters of the module and the attributes
        Inputs:
        Outputs:
        """
        self.inputs = []
        pass

    def zero_grad(self):
        """
        Goal:
        Reset the gradient tensors to zero
        Inputs:
        Outputs:
        """
        pass

    @property
    def params(self):
        """
        Goal:
        Return the parameters of the module and the associated gradients
        Inputs:
        Outputs:
        dict - dictionnary having as key the module instance itself and the value is a list of size 2
               the first element is the tensor corresponding to the parameters of the model, and the second
               is the associated gradient tensor. If the model is parameter-less then these last elements 
               are replaced by None
        """
        return {self:[None,None]}

#################### Linear class #################### 
class Linear(FrameworkModule):
    
    def __init__(self,in_size,out_size,add_bias=True):
        """
        Goal:
        Inputs:
        in_size = int > 0 - Input size of the layer
        out_size = int > 0 - Output size of the layer
        add_bias = bool - Specify if you want to add a bias or not
        Outputs:
        """
        super().__init__()
        xavier = math.sqrt(6/(in_size+out_size))
        if add_bias:
            # Initialize the weights and the associated gradient with Xavier Initialisation
            self.weights = empty(out_size,in_size+1).uniform_(-xavier,xavier)
        else: 
            # Initialize the weights and the associated gradient with Xavier Initialisation
            self.weights = empty(out_size,in_size).uniform_(-xavier,xavier)
        # Initialize the gradients to 0
        self.grdweights = empty(self.weights.shape).fill_(0)
        self.bias = add_bias
        # Initialize the inputs attribute
        self.inputs = []
            
    def forward(self,inputs,no_grad=False):
        """
        Goal:
        Perform the forward step
        Inputs:
        inputs = torch tensor - size NxDin (N number of datapoints, Din input size of the layer)
        no_grad = Boolean - Specify wheter or not you want to perform a backward step after
        Outputs:
        output = torch tensor - size NxDout (N number of datapoints, Dout output size of the layer)
        """
        # Compute the output
        if self.bias:
            inputs_ext = empty(inputs.shape[0],inputs.shape[1] + 1).fill_(1)
            inputs_ext[:,:-1] = inputs
        else:
            inputs_ext = inputs.clone()
        output = inputs_ext@(self.weights.T)
        # Store the input into the input attribute (will be useful for the backward step)
        if not no_grad:
            self.inputs.append(inputs_ext) 
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
        # Compute the gradient with respect to the input
        if self.bias:
            grdwrtinput = grdwrtoutput@(self.weights[:,:-1])
        else:
            grdwrtinput = grdwrtoutput@self.weights 
        # Compute the gradient with respect to the weights and accumulate
        self.grdweights += (grdwrtoutput.T)@inputs
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

    def update(self,new_params):
        """
        Goal:
        Replace the parameters of the model (weights and bias) by the new parameters
        Inputs:
        new_params = dict - dictionary having at least as key the module itself. The value associated
                     to this keep is a list containing the new parameters tensor
        Outputs:
        """
        self.weights = new_params[self]
        
    def reset(self):
        """
        Goal:
        Reset the weigths and bias, reset also the gradient tensors to 0, reset the inputs attribute
        Inputs:
        Outputs:
        """
        out_size, in_size = self.weights.shape # Get the shape of the layer
        if self.bias:
            in_size -= 1
        xavier = math.sqrt(6/(in_size+out_size)) # Xavier Initialization
        # Reset the weights
        self.weights = empty(self.weights.shape).uniform_(-xavier,xavier)
        # Set gradients to 0
        self.zero_grad()
        # Reset the inputs to None
        self.inputs = []
        
    @property
    def params(self):
        """
        Goal:
        Return the parameters of the module and the associated gradients
        Inputs:
        Outputs:
        dic - dictionnary having as key the module instance itself and the value is a list of size 2
              the first element is the tensor corresponding to the weights and bias of the module, and the second
              is the associated gradient tensor.
        """
        return {self:[self.weights,self.grdweights]}

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
        self.outputs = None
        # Intialize the targets attribute
        self.targets = None
       
    def forward(self,outputs,targets):
        """
        Goal:
        Compute the loss
        Inputs:
        inputs = torch tensor - size NxD (N number of datapoints, D dimension of the target)
        targets = torch tensor - size NxD (N number of datapoints, D dimension of the target)
        Outputs:
        loss = float - MSE loss 
        """
        # Store the inputs into the inputs attributes (will be useful for the backward)
        self.outputs = outputs
        # Store the targets into the inputs attributes (will be useful for the backward)
        self.targets = targets
        if self.mean:
            # Compute the loss and take the mean
            loss = ((outputs - targets)**2).mean()
        else:
            # Compute the loss and take the sum
            loss = ((outputs - targets)**2).sum()
        return loss
            
    def backward(self):
        """
        Goal:
        Compute the gradient of the loss with respect to the output of the last layer
        Inputs:
        Outputs:
        grdwrtoutput = torch tensor - size NxD (N number of data points, D dimension of the target)
        """
        # Raise a message error if the forward step has not been performed
        if self.outputs is None or self.targets is None:
            return "Forward step has not been performed"
        if self.mean:
            # Compute the gradient 
            grdwrtoutput = 2*(self.outputs-self.targets)/(self.outputs.shape[0])
        else:
            grdwrtoutput = 2*(self.outputs-self.targets)
        return grdwrtoutput    
    

#################### Activation functions class #################### 

class ActivationFunction(FrameworkModule):

    def __init__(self):
        """
        Goal:
        Inputs:
        Outputs: 
        """
        super().__init__()
        self.inputs = []

    def function(self,x):
        """
        Goal:
        Evaluate the activation function pointwise to the tensor x
        Inputs:
        x = torch.tensor - size NxD (N number of data points, D dimension of the input)
        Outputs: 
        tensor of same size 
        """
        return empty(1,1)

    def derivative(self,x):
        """
        Goal:
        Evaluate the first derivative of the activation function pointwise to the tensor x
        Inputs:
        x = torch.tensor - size NxD (N number of data points, D dimension of the input)
        Outputs: 
        tensor of same size 
        """
        return empty(1,1)

    def forward(self,inputs,no_grad=False):
        """
        Goal:
        Perform the forward step
        Inputs:
        inputs = torch tensor - size NxD (N number of data points, D dimension of the input)
        no_grad = Boolean - Specify wheter or not you want to perform a backward step after
        Outputs: 
        output = torch tensor - size NxD (N number of data points, D dimension of the input)
        """
        if not no_grad:
            self.inputs.append(inputs)
        output = self.function(inputs)
        return output

    def backward(self,grdwrtoutput):
        """
        Goal: 
        Compute the gradient with respect to the input
        Inputs: 
        grdwrtoutput = torch tensor - size NxD where D is the dimension of the layer and N the number of samples
        Outputs: 
        torch tensor of the same size storing the gradient with respect to the input
        """
        if len(self.inputs) == 0:
            return "Forward step has not been performed"
        inputs = self.inputs.pop()
        grdphi = self.derivative(inputs)
        grdwrtinput = grdphi*grdwrtoutput
        return grdwrtinput

#################### ReLU class #################### 

class ReLU(ActivationFunction):
    
    def __init__(self):
        """
        Goal:
        Inputs:
        Outputs: 
        """
        super().__init__()

    def function(self,x):
        """
        Goal:
        Evaluate the activation function pointwise to the tensor x
        Inputs:
        x = torch.tensor - size NxD (N number of data points, D dimension of the input)
        Outputs: 
        tensor of same size 
        """
        output = x.clone()
        output[x <= 0] = 0
        return output

    def derivative(self,x):
        """
        Goal:
        Evaluate the first derivative of the activation function pointwise to the tensor x
        Inputs:
        x = torch.tensor - size NxD (N number of data points, D dimension of the input)
        Outputs: 
        tensor of same size 
        """
        grdphi = empty(x.shape).fill_(0)
        grdphi[x > 0] = 1
        return grdphi

#################### Tanh class #################### 

class Tanh(ActivationFunction):
    
    def __init__(self):
        """
        Goal:
        Inputs:
        Outputs: 
        """
        super().__init__()
        
    def function(self,x):
        """
        Goal:
        Evaluate the activation function pointwise to the tensor x
        Inputs:
        x = torch.tensor - size NxD (N number of data points, D dimension of the input)
        Outputs: 
        tensor of same size 
        """
        return (x.exp() - (-x).exp())/(x.exp() + (-x).exp())

    def derivative(self,x):
        """
        Goal:
        Evaluate the first derivative of the activation function pointwise to the tensor x
        Inputs:
        x = torch.tensor - size NxD (N number of data points, D dimension of the input)
        Outputs: 
        tensor of same size 
        """
        return 1 - self.function(x)**2

#################### Sequential class #################### 

class Sequential(FrameworkModule):
    
    def __init__(self,*sequence):  # deleted the * before "sequence" so that a list of modules works
        """
        Goal:
        Inputs: 
        sequence = list of modules
        Outputs:
        """
        super().__init__()
        # Initialize the sequence attribute, containing modules of models and losses
        self.sequence = sequence
        
    def forward(self,inputs,no_grad=False):
        """
        Goal:
        Perform the forward path
        Inputs:
        inputs = list of modules
        no_grad = Boolean - Specify wheter or not you want to perform a backward step after
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

    def update(self,new_params):
        """
        Goal:
        Replace the parameters of the model by the new parameters
        Inputs:
        new_params = dict - dictionary having as key the modules of the sequence. The value associated
                     to these keys are tensors containing the new parameters 
        Outputs:
        """
        for module in new_params:
            module.update(new_params)
                
    def zero_grad(self):
        """
        Goal: 
        set all gradients to zero
        Inputs: 
        Outputs:
        """
        for module in self.sequence:
            # Use the method zero_grad of all involved modules, which fulfill that purpose
            module.zero_grad()

    def reset(self):
        """
        Goal: 
        reset all parameters to their default value, reset the gradients to 0, reset the inputs attribute
        Inputs: 
        Outputs:
        """
        for module in self.sequence:
            module.reset()
    
    @property
    def params(self):
        """
        Goal:
        Return the parameters of the module and the associated gradients
        Inputs:
        Outputs:
        dict - dictionnary having as key the module instances of the sequence and the values are list of size 2:
               the first element is the tensor corresponding to the parameters of the module, and the second
               is the associated gradient tensor.
        """
        # Initialize the list of all parameters
        dic_params = {}
        for module in self.sequence:
            dic_params = {**dic_params, **module.params}
        return dic_params

#################### Optimizer class #################### 
       
class Optim():

    def __init__(self,model,lr=1e-2):
        """
        Goal: 
        Inputs: 
        model = model to be optimized
        lr = float - learning rate
        Outputs:
        """
        self.model = model
        self.lr = lr

    def grad_descent(self,param,grdparam):
        """
        Goal: 
        Compute new parameters using gradient descent
        Inputs: 
        param = torch tensor - tensor containing parameters 
        grdparam = torch tensor - tensor containing corresponding gradients
        Outputs:
        """
        if param is None or grdparam is None:
            new_param = []
        else:
            new_param = param - self.lr*grdparam
        return new_param

    def optimize(self):
        """
        Goal: 
        Compute new parameters and pass these new parameters to the model
        Inputs: 
        Outputs:
        """
        params = self.model.params # Get the current parameters
        new_params = {}
        for mod_param in params:
            par = params[mod_param][0]
            grdpar = params[mod_param][1]
            new_params[mod_param] = self.grad_descent(par,grdpar) # Compute new parameters
        self.model.update(new_params)