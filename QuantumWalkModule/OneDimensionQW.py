import sys,os
sys.path.append(os.path.abspath(".."))

import torch
from QuantumWalkModule.Operator import Op
from QuantumWalkModule.StatesPreparation import States
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod

###########################################################################
# This class is desined for simple prototype of
# One dimensional random walk with PBCs.
# It implements the simplest and the most costful method 
# of simulating QRWs.
# The interface is desined in the way that
# allows user to run QRW with specific Coin operator
# chosen by the user

class QW_base(ABC):
    def __init__(self, dim : int, coin_conf : dict) -> None:
        self.dim = dim
        self.S = self.Step()
        self.C = self.Coin(**coin_conf)
        self.U = self.S @ self.C
        self.st_gen = States(self.dim)

    def Step(self) -> torch.tensor :
        right = torch.roll(torch.eye(self.dim),1,0)
        left = torch.roll(torch.eye(self.dim),-1,0)
        return torch.kron(right,Op.s_r) + torch.kron(left,Op.s_l)
    
    @abstractmethod
    def Coin(self,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def run(self,stpes,**kwars) -> torch.tensor:
        raise NotImplementedError
    
###########################################################################
# This abstract class extends the idea of QW with 
# evolution of pure state. 
# One should still implemet Coin operator
    
class QW_base_pure(QW_base,ABC):
    def run(self,steps,**kwargs) -> torch.tensor :
        st_type = kwargs.get("state", "Gaussian")
        mu = kwargs.get("mu",self.dim // 2)
        sigma = kwargs.get("sigma",self.dim / 40)
        state = self.st_gen.generate(st_type, mu = mu, sigma = sigma)
        data = self.st_gen.get_prob(state).unsqueeze(0)
        for _ in range(steps):
            state = self.U @ state
            new_data = self.st_gen.get_prob(state).unsqueeze(0)
            data = torch.concat((data,new_data),dim = 0)
        return data
    
###########################################################################
# This class implements the simple one - dimensional QW 
# With uniform coin and pure state evolution

class QW_1d(QW_base_pure):
    def Coin(self,**kwargs) -> torch.tensor:
        th = kwargs.get("th", np.pi / 4)
        E,U = torch.linalg.eig(Op.sy)
        coin = U @ torch.diag(torch.exp(-1j * E * th / 2)) @ U.conj().T
        return torch.kron(torch.eye(self.dim),coin)
    
###########################################################################
# This abstract class extends the idea of QW with 
# evolution of mixed state with dm state. 
# One should still implement Coin operator
    
class QW_base_mixed(QW_base,ABC):
    def run(self,steps,**kwargs) -> torch.tensor :
        # possible kwargs:
        # - state -> type of initial state
        # - state_conf -> dict of state parameters 
        # - kraus_conf -> dict of kraus parameters

        st_type = kwargs.get("state", "Gaussian")
        state_conf = kwargs.get("state_conf",{
            "mu" : self.dim // 2,
            "sigma" : self.dim / 40
        })
        kraus_conf = kwargs.get("kraus_conf",{
            "probs" : [1/3,1/3]
        })

        rho = self.st_gen.generate_dm(st_type, **state_conf)
        data = self.st_gen.get_prob_dm(rho).unsqueeze(0)
        Evo = self.get_Kraus(**kraus_conf)

        for _ in range(steps):
            rho = self.make_step(rho,Evo)
            new_data = self.st_gen.get_prob_dm(rho).unsqueeze(0)
            data = torch.concat((data,new_data),dim = 0)
        return data
    
    @abstractmethod
    def get_Kraus(self, **kwargs) -> list:
        raise NotImplementedError

    @abstractmethod
    def make_step(self, rho : torch.tensor,Evo : list) -> torch.tensor:
        raise NotImplementedError

###########################################################################
# This class implements the simple, one - dimensional QW 
# With uniform coin and mixed state in general case of Kraus 
# evolution

class QW_1d_dm_prototype(QW_base_mixed,ABC):
    def Coin(self,**kwargs) -> torch.tensor:
        th = kwargs.get("th", np.pi / 4)
        E,U = torch.linalg.eig(Op.sy)
        coin = U @ torch.diag(torch.exp(-1j * E * th / 2)) @ U.conj().T
        return torch.kron(torch.eye(self.dim),coin)
    
###########################################################################
# only for debugging purpose

"""
if __name__ == "__main__":
    dim = 50
    coin_conf = {
        "th" : np.pi/2
    }
    steps = 300
    model = QW_1d(dim , coin_conf)
    data = model.run(steps,state = "Gaussian")
    plt.imshow(data,origin = "lower")
    plt.show()
"""