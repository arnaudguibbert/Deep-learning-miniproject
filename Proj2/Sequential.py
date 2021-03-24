import torch
torch.set_grad_enabled(False)

class Sequential():
    
    def __init__(self,*sequence,loss=MSE):
        self.sequence = sequence
        self.inputs = None
        self.back = False
        
    def forward(self,inputs):
        self.inputs = inputs
        output = torch.clone(inputs)
        for module in self.sequence:
            output = module.forward(output)
        return output
        
    def backward(self,grdwrtoutput):
        if self.inputs == None:
            return "Forward pass has not been performed"
        for module in self.sequence:
            output = module.backward(grdwrtoutput)
            
    def optimization_step(self,lr):
        if not self.back:
            return "Backward step has not been performed"
        for module in self.sequence:
            if hasattr(module,"oprimization_step"):
                module.optimization_step(lr)
                
    def zero_grad(self):
    for module in self.sequence:
        if hasattr(module,"zero_grad"):
            module.zero_grad()

    def reset(self):
        for module in self.sequence:
            if hasattr(module,"reset"):
                module.reset()
    
    @property
    def params(self):
        params = []
        for module in self.sequence:
            mod_params = module.params
            if len(mod_params) > 0:
                params.append(mod_params)
        return params
       