# inspired by: https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/lenet5_pytorch.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),          #0
            #nn.ReLU(),                                                                  #1 a
            nn.Tanh(),                                                                  #1 b
            nn.AvgPool2d(kernel_size=2),                                                #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),         #3
            #nn.ReLU(),                                                                  #4 a
            nn.Tanh(),                                                                  #4 b
            nn.AvgPool2d(kernel_size=2),                                                #5
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),       #6
            #nn.ReLU(),                                                                   #7 a
            nn.Tanh(),                                                                  #7 b
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),            #0
            #nn.ReLU(),                                              #1 a
            nn.Tanh(),                                              #1 b
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
