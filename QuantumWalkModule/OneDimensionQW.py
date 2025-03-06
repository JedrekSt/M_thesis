import sys,os
sys.path.append(os.path.abspath(".."))

import torch
from QuantumWalkModule.Operator import Op
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC,abstractmethod

###########################################################################
# This class allows user to generate its own initial state
# depending on what distribution is the best for certain case

class States:
    def __init__(self,dim : int) -> None:
        self.dim = dim

    def Gaussian(self, sigma : float, mu : float, a : complex, b: complex) -> torch.tensor:
        x = torch.arange(0,self.dim,1)
        st = torch.exp( - (x - mu)**2 / (2 * sigma **2)) 
        c_st = torch.tensor([a,b])
        st = torch.kron(st,c_st)
        return st / torch.sqrt(st @ st.conj())
    
    def generate(self,name : str, **kwargs) -> torch.tensor:
        a = kwargs.get("a",1 )
        b = kwargs.get("b",1j)
        if name == "Gaussian":
            mu = kwargs.get("mu",self.dim // 2)
            sigma = kwargs.get("sigma",self.dim / 10)
            return self.Gaussian(sigma,mu,a,b)
        elif name == "Random":
            pass

    def get_prob(self,st : torch.tensor) -> torch.tensor:
        return np.abs(st[0::2])**2 + np.abs(st[1::2])**2

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
    
    def run(self,steps,**kwargs) -> torch.tensor :
        st_type = kwargs.get("state", "Gaussian")
        mu = kwargs.get("mu",self.dim // 2)
        sigma = kwargs.get("sigma",self.dim / 40)
        state = self.st_gen.generate(st_type,mu = mu, sigma = sigma)
        data = self.st_gen.get_prob(state).unsqueeze(0)
        for _ in range(steps):
            state = self.U @ state
            new_data = self.st_gen.get_prob(state).unsqueeze(0)
            data = torch.concat((data,new_data),dim = 0)
        return data

###########################################################################
# This class implements the abstract class of 
# simple QRW.

class QW_1d(QW_base):
    def Coin(self,**kwargs) -> torch.tensor:
        th = kwargs.get("theta", np.pi / 4)
        E,U = torch.linalg.eig(Op.sy)
        coin = U @ torch.diag(torch.exp(-1j * E * th / 2)) @ U.conj().T
        return torch.kron(torch.eye(self.dim),coin)

###########################################################################
# only for debugging purpose

"""if __name__ == "__main__":
    dim = 50
    coin_conf = {
        "th" : np.pi/2
    }
    steps = 300
    model = QW_1d(dim , coin_conf)
    data = model.run(steps,state = "Gaussian")
    plt.imshow(data,origin = "lower")
    plt.show()"""