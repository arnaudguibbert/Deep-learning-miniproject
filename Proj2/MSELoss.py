import torch
torch.set_grad_enabled(False)

class MSELoss():
    
    def __init__(self,mean=True):
        self.mean = mean
        self.inputs = None
        self.targets = None
       
    def forward(self,inputs,targets):
        self.inputs = inputs
        self.targets = targets
        if self.mean:
            loss = ((inputs - targets)**2).sum()/(targets.shape[0])
        else:
            loss = ((inputs - targets)**2).sum()
            
    def backward(self):
        if self.inputs is None or self.targets is None:
            return "Forward step has not been performed"
        if self.mean:
            grdwrtinput = 2*(self.inputs-self.targets).sum(dim=0)/self.targets.shape[0]
        else:
            grdwrtinput = 2*(self.inputs-self.targets).sum(dim=0)
        return grdwrtinput    
      
    @property
    def params(self):
        return []
        