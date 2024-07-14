##################
## Losses 
##################

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-6):
        inputs, targets = inputs.view(-1), targets.view(-1) # flatten label and prediction tensors
        intersection    = (inputs * targets).sum()                           
        dice_loss       = 1.0 - (2.0*intersection + eps)/(inputs.sum() + targets.sum() + eps)  
        bce             = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return (bce + dice_loss) / 2.0
        

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-6):
        inputs, targets = inputs.view(-1), targets.view(-1) # flatten label and prediction tensors
        intersection    = (inputs * targets).sum()                            
        dice_loss       = 1.0 - (2.0*intersection + eps)/(inputs.sum() + targets.sum() + eps)
        return dice_loss


class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, eps=1e-6):
        inputs, targets = inputs.view(-1), targets.view(-1) # flatten label and prediction tensors
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return bce_loss


class CombinedLoss(nn.Module):
    def __init__(self, weights=None, loss_modules=None):
        super().__init__()
        self.weights      = torch.from_numpy(weights).float()
        self.loss_modules = loss_modules

    def forward(self, inputs, targets, eps=1e-6):
        losses = []
        for loss_module in self.loss_modules:
            losses.append(loss_module(inputs, targets))
        loss = sum([w*l for w, l in zip(self.weights, losses)])
        return loss


class DeepSupervisionLoss(nn.Module):
    def __init__(self, weights=None, loss_modules=None):
        super().__init__()
        self.weights      = torch.from_numpy(weights).float()
        self.loss_modules = loss_modules

    def forward(self, inputs1, inputs2, inputs3, targets1, targets2, targets3, eps=1e-6):
        losses = []
        for loss_module in self.loss_modules:
            losses.append(loss_module(inputs1, targets1))
            losses.append(loss_module(inputs2, targets2))
            losses.append(loss_module(inputs3, targets3))
        loss = sum([w*l for w, l in zip(self.weights, losses)])
        return loss

