#!/usr/bin/env python

"""
Author: Vojtěch Čoupek
Description: Imagenet utilities
Project: Weight-Sharing of CNN - Diploma thesis FIT BUT 2023
Inspired by: - https://github.com/pytorch/examples/blob/master/imagenet/main.py
             - https://github.com/leaderj1001/MobileNetV3-Pytorch/blob/2cf3efa64d00c45e9ac61c3ef362396e9700fdb8/main.py#L117
"""

import torch
from torch.utils.data import DataLoader

# reference,
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
# Thank you.

class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val:float, n:int=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# reference,
# https://github.com/leaderj1001/MobileNetV3-Pytorch/blob/2cf3efa64d00c45e9ac61c3ef362396e9700fdb8/main.py#L117
# Thank you!

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].sum(dim=(0,1)).float()
            res.append(correct_k.mul_(1 / batch_size))
        return res

def get_accuracy(model:torch.nn.Module, data_loader:DataLoader, device:str='cpu', topk:int=1) -> float:
    """Function for computing the accuracy of the predictions over the entire data_loader.

    The top k means that if the correct solution is among the top k (k means number) nets guesses,
    its counted as correctly guessed. 

    Args:
        model (torch.nn.Module): Is the model to test the accuracy on.
        data_loader (DataLoader): Is the data loader to test the models accuracy on.
        device (str, optional): Is the device used for the calculation. Defaults to 'cpu'.
        topk (int, optional): Is the top k accuracy setting. Defaults to 1.

    Returns:
        float: the accuracy percentage.
    """

    top = AverageMeter(f'Acc@{topk}', ':6.2f')

    model.eval()
    with torch.no_grad():
        for data, target in data_loader:
            # if args.gpu is not None:
            #     input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)

            # measure accuracy and record loss
            acc = accuracy(output, target, topk=(topk,))
            top.update(acc[0], data.size(0))

    return top.avg.cpu().numpy().item(0)
