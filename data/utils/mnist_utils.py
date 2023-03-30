# inspired by: https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb
# for Le-Net

import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import copy
from torch.utils.data import DataLoader

def train_net(net:torch.nn.Module, criterion:torch.nn.modules.loss, optimizer:torch.optim, dataset, epochs:int, 
    device:str = None, print_every:int = 1, plot:bool = False):
    """Training loop for the net. The training sets the net to the best validation loss weights.

    Args:
        net (torch.nn.Module): Is the model to be trained.
        criterion (torch.nn.modules.loss): Is the loss function for the model.
        optimizer (torch.optim): Is the training optimizer.
        dataset (Dataset): Is the dataset to be trained on.
        epochs (int): Number of training epochs.
        device (str, optional): Is the device to by trained on. Defaults to None (tries 'cuda', if not available 'cpu').
        print_every (int, optional): Defines how often are the training details printed out. Defaults to 1.
        plot (bool, optional): If true, loss plost are shown. Defaults to False.
    """

    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    best_model = None

    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train model
    for epoch in range(1, epochs+1):

        epoch_start = time.time()

        net.train()
        running_loss = 0

        for X, y_true in dataset.train_dl:

            # Forward pass
            X , y_true = X.to(device), y_true.to(device)
            y_hat, _ = net(X) 
            loss = criterion(y_hat, y_true) 
            running_loss += loss.item() * X.size(0)
            
            optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        epoch_loss = running_loss / len(dataset.train_dl.dataset)
        train_losses.append(epoch_loss)

        # validation
        with torch.no_grad():
            valid_loss = validate(net, dataset.valid_dl, criterion, device)
            
            # saving the bes model on valid data
            if best_model is None or valid_loss < min(valid_losses):
                best_model = copy.deepcopy(net.state_dict())

            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(net, dataset.train_dl, device)
            valid_acc = get_accuracy(net, dataset.valid_dl, device)
            duration = time.time() - epoch_start
                
            print(
                f'Epoch: {epoch}/{epochs} --- '
                f'Time: {duration:.2f}s\t'
                f'Train loss: {epoch_loss:.4f}\t'
                f'Valid loss: {valid_loss:.4f}\t'
                f'Train accuracy: {100 * train_acc:.2f}%\t'
                f'Valid accuracy: {100 * valid_acc:.2f}%'
            )

    net.load_state_dict(best_model)

    train_acc = get_accuracy(net, dataset.train_dl, device)
    valid_acc = get_accuracy(net, dataset.valid_dl, device)
    test_acc = get_accuracy(net, dataset.test_dl, device)

    print(
        'Train completed --- ',
        f'Train accuracy: {100 * train_acc:.2f}%\t',
        f'Valid accuracy: {100 * valid_acc:.2f}%\t',
        f'Test accuracy: {100 * test_acc:.2f}%'
    )

    if plot:
        plot_losses(train_losses, valid_losses)
    
    #return optimizer, (train_losses, valid_losses), test_acc

def get_trained(net:torch.nn.Module, path:str, train_args:list, device:str = 'cpu') -> None:
    """Makes sure that the given net has trained values either by traning or loading given
    save file. If trained, the values are saved for next utilization.

    Args:
        net (torch.nn.Module): the model to ensure the loaded values in.
        path (str): is the save file path. If it not exists, the net is trained and the file is created.
        train_args (list): is the train settings for the net.
        device (str, optional): is the device to train on. Defaults to 'cpu'.
    """

    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_net(net, *train_args)
    torch.save(net.state_dict(), path)


def plot_losses(train_losses:list, valid_losses:list) -> None:
    """Function for plotting training and validation losses

    Args:
        train_losses (list): Are the training losses to be plotted.
        valid_losses (list): Are the valid losses to be plotted
    """
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    # fig.show()
    fig.savefig('lenet_tanh_train.pdf')
    
    # change the plot style to default
    plt.style.use('default')

def get_accuracy(net:torch.nn.Module, data_loader:DataLoader, device:str = 'cpu') -> float:
    """Function for computing the accuracy of the predictions over the entire data_loader

    Args:
        net (torch.nn.Module): Is the net to test the accuracy on.
        data_loader (Dataloader): Is the data dataloader to be the model tested on.
        device (str, optional): Is the device which the computation is performed on. Defaults to 'cpu'.

    Returns:
        float: Is the models accuracy on the data.
    """
        
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        net.eval()
        for X, y_true in data_loader:

            X = X.to(device)

            if isinstance(y_true, int):
                y_true = torch.tensor(y_true)

            y_true = y_true.to(device)

            _, y_prob = net(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.cpu().numpy() / n


def validate(net:torch.nn.Module, valid_loader:DataLoader, criterion:torch.nn.modules.loss, device:str = 'cpu') -> float:
    """Function for the validation step of the training loop

    Args:
        net (torch.nn.Module): Is the model to be validated
        valid_loader (Dataloader): is the validation data dataloader
        criterion (torch.nn.modules.loss): is the nets loss function.
        device (str, optional): Is the device to be computed on. Defaults to 'cpu'.

    Returns:
        float: the validation loss.
    """

    net.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = net(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return epoch_loss