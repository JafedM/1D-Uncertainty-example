from torch.utils.data import Dataset
import torch
import seaborn as sns

import matplotlib.pyplot as plt

def plot_uncertanity(x_plot, y_mean, y_std, title='Uncertanity'):
    '''
    Plot the uncertanity

    x_plot: Values in x
    y_mean: Predictions mean
    y_std:  Predictions std
    title:  Titulo de la grafica
    '''
    #clrs = sns.color_palette("husl", 5)
    with sns.axes_style("darkgrid"):
        x_plot = torch.flatten(x_plot)
        plt.plot(x_plot, y_mean, c='black')
        plt.fill_between(x_plot, y_mean-y_std, y_mean+y_std ,alpha=0.3,color='red')
        plt.fill_between(x_plot, y_mean-3*y_std, y_mean+3*y_std ,alpha=0.3,color='m')
        plt.title(title)
        plt.show()
        

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
@torch.inference_mode()
def eval_model(model, val_dataloader, criterion, epoch, device):
    batch_counter = 0
    val_loss = 0
    torch.cuda.empty_cache()
    model.eval()
    for input, tgt in val_dataloader:
        input = input.to(device)
        y_real = tgt.to(device)
        y_pred = model(input)
        val_loss += criterion(y_pred, y_real)
        batch_counter += 1
        
    val_loss = val_loss/batch_counter
    return val_loss

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
        torch.cuda.empty_cache()
        val_loss = 0 
        train_loss = 0
        batch_counter = 0

        model.train()
        for batch in train_dataloader:
            x, y_real = batch
            x, y_real = x.to(device), y_real.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y_real)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_counter += 1
            train_loss += loss
            
        train_loss = train_loss/batch_counter
        batch_counter = 0

        if (epoch+1) % verbose == 0:
            val_loss = eval_model(model, val_dataloader, criterion, epoch, device)
        
            print("Epoch: {} :::: Train loss {} :::: Val loss {}\n".format(epoch+1, train_loss, val_loss))