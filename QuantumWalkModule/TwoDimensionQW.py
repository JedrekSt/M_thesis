import numpy as np
import matplotlib.pyplot as plt
import torch
from abc import ABC, abstractmethod

from scipy.linalg import expm

class Walk_prototype(ABC):
  def __init__(self,dimx,dimy,x_coin_dict,y_coin_dict,initial_state_dict,device = "cpu"):
    self.dx = dimx
    self.dy = dimy

    self.device = device

    self.Coin1 = self.Coin(**x_coin_dict).to(self.device)
    assert isinstance(self.Coin1,torch.Tensor) and self.Coin1.shape == (self.dy,self.dx,4), "invalid Coin shape or type"
    self.Coin2 = self.Coin(**y_coin_dict).to(self.device)
    assert isinstance(self.Coin2,torch.Tensor) and self.Coin2.shape == (self.dy,self.dx,4), "invalid Coin shape or type"

    self.state = self.Initial_state(**initial_state_dict).to(self.device)
    assert isinstance(self.state,torch.Tensor) and self.state.shape == (self.dy,self.dx,2), "invalid state shape or type"

  @abstractmethod
  def Coin(self,**kwargs):
    raise NotImplementedError

  @abstractmethod
  def Initial_state(self,**kwargs):
    raise NotImplementedError

  def Evolve(self):
    # we assume that the Coin operator's shape is (dx,dy,4)
    # State vector is described by the tensor (dx,dy,2)
    self.state = self.Coin2[:,:,0::2].roll(1,0) * self.state[:,:,0:1].roll(1,0) + self.Coin2[:,:,1::2].roll(-1,0) * self.state[:,:,1:2].roll(-1,0)
    self.state = self.Coin1[:,:,0::2].roll(1,1) * self.state[:,:,0:1].roll(1,1) + self.Coin1[:,:,1::2].roll(-1,1) * self.state[:,:,1:2].roll(-1,1)

  def Get_prob(self):
    return (torch.abs(self.state[:,:,0])**2 + torch.abs(self.state[:,:,1])**2).to("cpu")


class SimpleWalk2D(Walk_prototype):
  def Coin(self,**kwargs):
    th = kwargs.get("th")

    rot = torch.tensor([
        [np.sin(th/2) , -np.cos(th/2)],
        [np.cos(th/2) ,  np.sin(th/2)]
    ],dtype = torch.complex64).flatten().reshape(1,1,4).to(self.device)

    Id = torch.ones(size = (self.dy,self.dx), dtype = torch.complex64)
    Id = Id.unsqueeze(2).to(self.device)
    return Id * rot

  def Initial_state(self, **kwargs):
    st = torch.zeros(size = (self.dy,self.dx), dtype = torch.complex64)
    st[self.dx // 2, self.dy // 2] = 1
    st = st.unsqueeze(2).to(self.device)

    a = kwargs.get("a",1)
    b = kwargs.get("b",1j)

    cs = torch.tensor([a,b],dtype = torch.complex64)
    cs /= np.sqrt((cs * cs.conj()).sum())
    cs = cs.unsqueeze(0).unsqueeze(0).to(self.device)
    return st * cs

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"currently used device : {device}")