from torch import empty

"""
Update 11/04:
added self.back = True at the end of backward
added self.back = False at the end of optimization_step
backward step implementation a bit strange
"""

class Sequential():
    
    def __init__(self,*sequence,loss="MSE"):
        """
        Goal:
        Inputs: 
        sequence = list of modules
        loss = module of a loss
        Outputs:
        """
        # Initialize the sequence attribute, containing modules of models and losses
        self.sequence = sequence
        # Initialize the inputs attribute
        self.inputs = None
        # Initialize the back flag (true if backpropagation has been performed)
        self.back = False
        
    def forward(self,inputs, no_grad=False):
        """
        Goal:
        Perform the forward path
        Inputs:
        inputs = list of modules
        Outputs:
        output of the forward path = torch tensor
        """
        self.inputs = inputs
        # Initialize the variable that will later contain the output
        output = inputs.clone()
        # Perform the forward path by calling the forward method of every module in the sequence
        for module in self.sequence:
            output = module.forward(output)
        return output
        
    def backward(self,grdwrtoutput):
        """
        Goal:
        Perform the backward path by updating the gradients
        Inputs:
        grdwrtoutput = torch tensor of size number of samples x size of the last layer // gradient with respect to the output
        Outputs:
        """
        if self.inputs == None:
            # Print an error message if forward hasn't been performed yet
            return "Forward pass has not been performed"
        for module in reversed(self.sequence):
            # Compute the gradients
            grdwrtoutput = module.backward(grdwrtoutput)
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
       