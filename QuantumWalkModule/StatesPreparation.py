import sys,os
sys.path.append(os.path.abspath(".."))

import torch
import numpy as np

###########################################################################
# This class allows user to generate its own initial state
# depending on what distribution is the best for certain case

class States:
    def __init__(self,dim : int) -> None:
        self.dim = dim

    def Gaussian(self, sigma : float, mu : float, a : complex, b: complex) -> torch.tensor:
        x = torch.arange(0,self.dim,1)
        st = torch.exp( - (x - mu) ** 2 / (2 * sigma ** 2)) 
        c_st = torch.tensor([a,b])
        st = torch.kron(st,c_st)
        return st / torch.sqrt(st @ st.conj())
    
    def Point(self,x0 : int, a : complex, b : complex) -> torch.tensor:
        st = torch.zeros((self.dim,))
        st[x0 % self.dim] = 1
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
        elif name == "Point":
            x0 = kwargs.get("x0",self.dim // 2)
            return self.Point(x0,a,b)
        else:
            raise NotImplementedError

    def get_prob(self,st : torch.tensor) -> torch.tensor:
        return st[0::2].abs()**2 + st[1::2].abs()**2
    
    def get_prob_dm(self,rho : torch.tensor) -> torch.tensor:
        st = torch.diag(rho)
        return (st[0::2] + st[1::2]).real
    
    def generate_dm(self,name : str , **kwargs) -> torch.tensor:
        st = self.generate(name,**kwargs)
        return torch.kron(st,st.conj().reshape(-1,1))
