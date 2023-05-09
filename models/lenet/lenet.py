#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Implementation of Le-Net-5 CNN
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
Inspired by: https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, n_classes:int, type:str='tanh'):
        """Inits the Le-Net5 model.

        Args:
            n_classes (int): Number of output neurons.
            type (str, optional): Net type - tanh or relu. Defaults to 'tanh'.

        Raises:
            Exception: if unknoun net type specified.
        """

        if type not in ['relu', 'tanh']:
            raise Exception('Unknown net type')
        
        super(LeNet5, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),          #0
            nn.ReLU() if type == 'relu' else nn.Tanh(),                                 #1
            nn.AvgPool2d(kernel_size=2),                                                #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),         #3
            nn.ReLU() if type == 'relu' else nn.Tanh(),                                 #4
            nn.AvgPool2d(kernel_size=2),                                                #5
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),       #6
            nn.ReLU() if type == 'relu' else nn.Tanh(),                                 #7
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),            #0
            nn.ReLU() if type == 'relu' else nn.Tanh(),             #1
            nn.Linear(in_features=84, out_features=n_classes),      #2
        )

    def forward(self, x):
        
        x = self.quant(x)

        x = self.feature_extractor(x)
        
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        
        logits = self.dequant(logits)

        probs = F.softmax(logits, dim=1)

        return logits, probs
