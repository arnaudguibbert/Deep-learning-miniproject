import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self,layer_size=16):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(2, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,layer_size),
            nn.ReLU(),
            nn.Linear(layer_size,1),
            nn.Tanh()
        )

    def forward(self,input):
        output = self.sequence(input)
        return output

def train_pytorch_model(model,train_inputs,train_targets,
                        epochs=100,mini_batch_size=100,lr=5e-2):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        for b in range(0,train_inputs.shape[0],mini_batch_size):
            inputs = train_inputs.narrow(0, b, mini_batch_size)
            output = model(inputs)
            target = train_targets.narrow(0, b, mini_batch_size)
            loss = criterion(output,target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()