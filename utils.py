from torch.utils.data import Dataset
import torch

import matplotlib.pyplot as plt

def plot_uncertanity(x_plot, y_mean, y_std):
    '''
    Plot the uncertanity

    x_plot: Values in x
    y_mean: Prediccions mean
    y_std:  Prediccions std
    '''
    x_plot = torch.flatten(x_plot)
    plt.plot(x_plot, y_mean, c='black')
    plt.fill_between(x_plot, y_mean-y_std, y_mean+y_std ,alpha=0.3,color='red')
    plt.title("Deep ensemble uncertanity")
    plt.grid()

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def trainer(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=10, device=None, verbose=5):
    '''
    Function to train a model
    '''
    #Device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    model.to(device)

    #Training
    for epoch in range(epochs):
        val_loss = 0 
        train_loss = 0
        batch_counter = 0

        model.train()
        for batch in train_dataloader:
            x, y_real = batch
            x, y_real = x.to(device), y_real.to(device)

            y_pred = model(x)

            loss = criterion(y_real, y_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_counter += 1
            train_loss += loss
            
        train_loss = train_loss/batch_counter
        batch_counter = 0

        model.eval()
        for batch in val_dataloader:
            x, y_real = batch
            x, y_real = x.to(device), y_real.to(device)

            y_pred = model(x)

            loss = criterion(y_real, y_pred)

            batch_counter += 1
            val_loss += loss
            
        val_loss = val_loss/batch_counter

        model.eval()
        
        if epoch%verbose == 0:
            print("Epoch: {} :::: Train loss {} :::: Val loss {} \n".format(epoch, train_loss, val_loss))