#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Net accuracy and compression predictor
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
"""

import torch
import time
import matplotlib.pyplot as plt
import copy
import numpy as np
import os

class NetPerfPredict(torch.nn.Module):
    
    def __init__(self, feature_pass_list:list):
        super(NetPerfPredict, self).__init__()

        self.features = []

        def lin_layer(inp, oup):
            return torch.nn.Sequential(
                torch.nn.Linear(inp, oup),
                torch.nn.Tanh()
            )

        for i in range(len(feature_pass_list) - 1):
            self.features.append(lin_layer(feature_pass_list[i], feature_pass_list[i+1]))
        
        self.features = torch.nn.Sequential(*self.features)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(feature_pass_list[-1], 1),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        
        return self.output(x)
    
def train_net(net:torch.nn.Module, criterion, optimizer, dataset, epochs, device=None, print_every=1, plot=False):
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
                y_hat = net(X)  
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
                    f'Train accuracy: {train_acc}\t'
                    f'Valid accuracy: {valid_acc}'
                )

        net.load_state_dict(best_model)

        train_acc = get_accuracy(net, dataset.train_dl, device)
        valid_acc = get_accuracy(net, dataset.valid_dl, device)
        test_acc = get_accuracy(net, dataset.test_dl, device)

        print(
            'Train completed --- ',
            f'Train accuracy: {train_acc}\t',
            f'Valid accuracy: {valid_acc}\t',
            f'Test accuracy: {test_acc}'
        )

        if plot:
            plot_losses(train_losses, valid_losses)
        
        #return optimizer, (train_losses, valid_losses), test_acc

def get_trained(net:torch.nn.Module, path:str, train_args:list, device:str='cpu') -> None:

    if os.path.exists(path):
        net.load_state_dict(torch.load(path, map_location=device))
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    train_net(net, *train_args)
    torch.save(net.state_dict(), path)


def plot_losses(train_losses:list, valid_losses:list):
    '''
    Function for plotting training and validation losses
    '''
    
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

def get_accuracy(net:torch.nn.Module, data_loader, device):
        '''
        Function for computing the accuracy of the predictions over the entire data_loader
        '''
        
        deltas = 0
        n = 0
        
        with torch.no_grad():
            net.eval()
            for X, y_true in data_loader:

                X = X.to(device)

                if isinstance(y_true, int):
                    y_true = torch.tensor(y_true)

                y_true = y_true.to(device)

                out = net(X)

                n += y_true.size(0)
                deltas = max(np.max(torch.abs(torch.sub(y_true, out)).cpu().numpy())[0], deltas)

        return deltas / n


def validate(net:torch.nn.Module, valid_loader, criterion, device):
        '''
        Function for the validation step of the training loop
        '''
    
        net.eval()
        running_loss = 0
        
        for X, y_true in valid_loader:
        
            X = X.to(device)
            y_true = y_true.to(device)

            # Forward pass and record loss
            y_hat = net(X) 
            loss = criterion(y_hat, y_true) 
            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(valid_loader.dataset)
            
        return epoch_loss