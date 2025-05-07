import numpy as np
import torch
from abc import ABC, abstractmethod

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

class SimpleWalk2D(Walk_prototype, ABC):
  def Coin(self,**kwargs):
    th = kwargs.get("th")

    rot = torch.tensor([
        [np.sin(th/2) , -np.cos(th/2)],
        [np.cos(th/2) ,  np.sin(th/2)]
    ],dtype = torch.complex64).flatten().reshape(1,1,4).to(self.device)

    Id = torch.ones(size = (self.dy,self.dx), dtype = torch.complex64)
    Id = Id.unsqueeze(2).to(self.device)
    return Id * rot
  
class MagneticWalk(SimpleWalk2D, ABC):
  def __init__(self, dimx, dimy, x_coin_dict, y_coin_dict, initial_state_dict, B , device = "cpu") -> None:
    super().__init__(dimx, dimy, x_coin_dict, y_coin_dict, initial_state_dict, device = device)
    self.F = self.Magnetic_phase(B)

  def Magnetic_phase(self, B: float) -> torch.tensor:
    X_ = torch.arange(self.dx).reshape(self.dx,1,1)
    F_ = torch.exp(1j * B * X_)
    F_ = torch.concat((F_,torch.zeros_like(X_)), dim = 2)
    F_ = torch.concat((F_,torch.zeros_like(X_)), dim = 2)
    F_ = torch.concat((F_,torch.exp(-1j * B * X_)), dim = 2)
    F_final = F_.clone()
    for _ in range(self.dy-1):
      F_final = torch.concat((F_final, F_) , dim = 1)
    return F_final.to(self.device)
  
  def Evolve(self):
    super().Evolve()
    self.state = self.F[:,:,0::2] * self.state[:,:,0:1] + self.F[:,:,1::2] * self.state[:,:,1:2]