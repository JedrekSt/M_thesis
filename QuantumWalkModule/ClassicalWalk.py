import sys,os
sys.path.append(os.path.abspath(".."))

import torch

###########################################################################

class ClWalk:
    def __init__(self,dim : int) -> None:
        self.dim = dim 

    def Evo(self,p : float) -> torch.tensor:
        left = torch.roll(torch.eye(self.dim),1,0)
        right = torch.roll(torch.eye(self.dim),-1,0)
        return p * left + (1-p) * right
    
    def run(self,p : float,steps : int) -> torch.tensor:
        U = self.Evo(p)
        st = torch.zeros(self.dim)
        data = st.unsqueeze(0)
        st[self.dim //2] = 1
        for _ in range(steps):
            st = U @ st
            data = torch.concatenate((data,st.unsqueeze(0)),dim = 0)
        return data
