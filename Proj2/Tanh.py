import torch
torch.set_grad_enabled(False)

class Tanh():
    
    def __init__(self):
        self.inputs = []
        
    def tanh(self,x):
        return (torch.exp(x) - torch.exp(-x))/(torch.exp(x) + torch.exp(-x))
        
    def forward(self,inputs):
        self.inputs.append(inputs)
        output = self.tanh(inputs)
        return output
        
    def backward(self,grdwrtoutput):
        if len(self.inputs) == 0:
            "Forward step has not been performed"
        inputs = self.inputs.pop()
        grdwrtinput = (1 - self.tanh(inputs)**2)*grdwrtoutput
        return grdwrtinput
        
    @property
    def params(self):
        return []