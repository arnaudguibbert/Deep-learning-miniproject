import torch
import torch.nn as nn

def train_model(model, train_input, train_target, nb_epochs, mini_batch_size = 100, eta = 0.1, criterion = nn.CrossEntropyLoss()):
    
    optimizer = torch.optim.SGD(model.parameters(), lr = eta)
    
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
     
            
def compute_nb_errors(model, data_input, data_target, mini_batch_size=100):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors