import torch
torch.set_grad_enabled(False)

class ReLU():
    
    def __init__(self):
        self.inputs = []
        
    def forward(self,inputs):
        self.inputs.append(inputs)
        output = torch.clone(inputs)
        output[inputs <= 0] = 0
        return output
        
    def backward(self,grdwrtoutput):
        inputs = self.inputs.pop(0)
        grdphi = torch.zeros_like(inputs)
        grdphi[inputs > 0] = 1
        grdwrtinput = grdphi*grdwrtoutput
        return grdwrtinput
        
    @property
    def params(self):
        return []