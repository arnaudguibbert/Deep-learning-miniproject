from torch import empty

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