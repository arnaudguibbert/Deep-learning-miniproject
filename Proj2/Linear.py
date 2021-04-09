from torch import empty

#tutorial

class Linear():
    
    def __init__(self,in_size,out_size,add_bias=True):
        """
        Goal:
        Inputs:
        in_size = int - Input size of the layer
        out_size = int - Output size of the layer
        add_bias = bool - Specify if you want to add a bias or not
        Outputs:
        """
        # Initialize the weights and the associated gradient
        self.weights = empty(out_size,in_size).normal_(mean=0,std=2) # To be changed
        self.grdweights = empty(out_size,in_size).fill_(0)
        if add_bias: 
            # Initialize the bias
            self.bias = empty(out_size,1).normal_(mean=0,std=2)
            self.grdbias = empty(out_size,1).fill_(0)
        else:
            self.bias = None
            self.grdbias = None
        self.inputs = None
        self.back = False
            
    def forward(self,inputs):
        self.inputs = inputs
        output = inputs@(self.weights.T)
        if self.bias is not None:
            output += self.bias
        return output
        
    def backward(self,grdwrtoutput):
        if self.inputs is None:
            return "Forward step has not been performed"
        grdwrtinput = grdwrtoutput@self.weights
        self.grdweights += (grdwrtoutput.T)@self.inputs
        if self.bias is not None:
            self.grdbias += grdwrtoutput
        self.back = True
        return grdwrtinput
    
    def zero_grad(self):
        self.grdweights = empty(self.weights.shape).fill_(0)
        if self.bias is not None:           
            self.grdbias = empty(self.bias.shape).fill_(0)
        
    def optimization_step(self,lr):
        if not self.back:
            return "Backward step has not been performed"
        self.weights -= self.grdweights*lr
        if self.bias is not None:
            self.bias -= self.grdbias*lr
        self.back = False
        
    def reset(self):
        self.weights = torch.empty(in_size,out_size).normal_(mean=0,std=2)
        if self.bias is not None:
            self.bias = torch.empty(out_size,1)
        self.zero_grad()
        self.inputs = None
        
    @property
    def params(self):
        if self.bias is None:
            param = self.weights
            grdparam = self.grdweights
        else:
            param = torch.cat((self.weights,self.bias),dim=1)
            grdparam = torch.cat((self.grdweights,self.grdbias),dim=1)
        return param, grdparam
    
        