from models.unet.unet_model import UNET
from data.oxfordPetDataset import OxfordPetDataset
import torch

dataset = OxfordPetDataset('./data')
net = UNET(3, 1)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=0.01)


def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.to(torch.device('cpu'))).float().mean()

net.train_model(10, dataset, opt, loss_fn, acc_metric)
