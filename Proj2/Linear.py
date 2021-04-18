from torch import empty
import torch
import math


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
            param = torch.cat((self.weights,self.bias),dim=1) # To be modified
            grdparam = torch.cat((self.grdweights,self.grdbias),dim=1) # To be modified
        return param, grdparam
    
        