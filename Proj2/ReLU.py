from torch import empty

class ReLU():
    
    def __init__(self):
        self.inputs = []
        
    def forward(self,inputs):
        self.inputs.append(inputs)
        output = inputs.clone()
        output[inputs <= 0] = 0
        return output
        
    def backward(self,grdwrtoutput):
        inputs = self.inputs.pop(0)
        grdphi = empty(inputs.shape).fill_(0)
        grdphi[inputs > 0] = 1
        grdwrtinput = grdphi*grdwrtoutput
        return grdwrtinput
        
    @property
    def params(self):
        return []