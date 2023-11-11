import torch
from torch import nn

from masksembles.torch import Masksembles1D

class MLP(nn.Module):
  '''
    Regression MLP

    acu_uncertainty:  Bool to Apply aleatoric uncertainty
  '''
  def __init__(self, acu_uncertainty=False):
    super().__init__()
    self.linear_1 = nn.Linear(1, 64)
    self.act_1 = nn.ReLU()
    if acu_uncertainty:
      self.linear_2 = nn.Linear(64, 2)
    else:
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

    N:                N mask
    s:                Masksembles rate
    acu_uncertainty:  Bool to Apply aleatoric uncertainty
  '''
  def __init__(self, N=4, s=2.0, acu_uncertainty=False):
    super().__init__()
    self.linear_1 = nn.Linear(1, 64)
    self.mask_1 = Masksembles1D(64, N, s)
    self.act_1 = nn.ReLU()
    if acu_uncertainty:
      self.linear_2 = nn.Linear(64, 2)
    else:
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
    
    p:                Dropout rate
    acu_uncertainty:  Bool to Apply aleatoric uncertainty
  '''
  def __init__(self, p=0.1, acu_uncertainty=False):
    super().__init__()
    self.linear_1 = nn.Linear(1, 64)
    self.drop_1 = nn.Dropout(p)
    self.act_1 = nn.ReLU()
    if acu_uncertainty:
      self.linear_2 = nn.Linear(64, 2)
    else:
      self.linear_2 = nn.Linear(64, 1)
      
  def forward(self, x):
    '''
      Forward pass
    '''
    x = self.linear_1(x)
    x = self.drop_1(x)
    x = self.act_1(x)
    return self.linear_2(x)
  
