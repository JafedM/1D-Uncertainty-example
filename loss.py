import torch

def acu_loss(output, target):
    ''''
    Computes the Aleatoric uncertainty loss

    output: Model output with value and variance
    target: GT
    '''
    loss = torch.mean(0.5*torch.exp(-output[:,1]) * (target.view(len(target)) - output[:,0])**2 + 0.5*output[:,1]) 
    return loss