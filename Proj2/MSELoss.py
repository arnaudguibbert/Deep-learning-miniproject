from torch import empty


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
            loss = (((inputs - targets)**2).sum(dim=1)).mean()
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
            grdwrtinput = 2*(self.inputs-self.targets).mean(dim=0)
        else:
            grdwrtinput = 2*(self.inputs-self.targets).sum(dim=0)
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
        