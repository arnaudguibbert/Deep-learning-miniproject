from torch import empty

class ReLU():
    
    def __init__(self):
        """
        Goal:
        Inputs:
        Outputs: 
        """
        # Initialize the list of inputs attribute
        self.inputs = []
        
    def forward(self,inputs):
        """
        Goal:
        Perform the forward step using ReLU
        Inputs:
        inputs = torch tensor
        Outputs: 
        torch tensor of the same size
        """
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
        inputs = self.inputs.pop(0)
        
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