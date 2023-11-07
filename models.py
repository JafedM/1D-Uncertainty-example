import torch
from torch import nn

from masksembles.torch import Masksembles1D

class MLP(nn.Module):
  '''
    Regression MLP
  '''
  def __init__(self):
    super().__init__()
    self.linear_1 = nn.Linear(1, 64)
    self.act_1 = nn.ReLU()
    self.linear_2 = nn.Linear(64, 1)
  def forward(self, x):
    '''
      Forward pass
    '''
    x = self.linear_1(x)
    x = self.act_1(x)
    return self.linear_2(x)
  
class MLP_Maksembles(nn.Module):
  '''
    Regression MLP with Maksembles
  '''
  def __init__(self, N=4, s=2.0):
    super().__init__()
    self.linear_1 = nn.Linear(1, 64)
    self.mask_1 = Masksembles1D(64, N, s)
    self.act_1 = nn.ReLU()
    self.linear_2 = nn.Linear(64, 1)
  def forward(self, x):
    '''
      Forward pass
    '''
    x = self.linear_1(x)
    x = self.mask_1(x).float()
    x = self.act_1(x)
    return self.linear_2(x)
  
class MLP_Dropout(nn.Module):
  '''
    Regression MLP with dropout
  '''
  def __init__(self):
    super().__init__()
    self.linear_1 = nn.Linear(1, 64)
    self.drop_1 = nn.Dropout(0.1)
    self.act_1 = nn.ReLU()
    self.linear_2 = nn.Linear(64, 1)
  def forward(self, x):
    '''
      Forward pass
    '''
    x = self.linear_1(x)
    x = self.drop_1(x)
    x = self.act_1(x)
    return self.linear_2(x)
  
