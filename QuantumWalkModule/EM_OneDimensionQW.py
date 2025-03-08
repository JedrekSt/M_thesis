import sys,os
sys.path.append(os.path.abspath(".."))

import torch
from QuantumWalkModule.OneDimensionQW import QW_1d_uniform_coined

###########################################################################
# This class implements the QW with 
# External electric field given by 
# phase change.
# field_conf :
# - name -> unfiform, ...
# - ratio -> ratio: phi / 2 pi

class Electric_QW_1d(QW_1d_uniform_coined):
    def __init__(self, dim : int, coin_conf : dict, field_conf: dict) -> None:
        super().__init__(dim,coin_conf)
        self.Phi = self.Field_operator(**field_conf)
        self.U = self.Phi @ self.U
    
    def make_step(self, state : torch.tensor) -> torch.tensor:
        return self.U @ state
    
    def Field_operator(self,**kwargs) -> torch.tensor:
        name = kwargs.get("name", "uniform")
        if name == "uniform":
            ratio = kwargs.get("ratio",1)
            x_ = torch.arange(0,self.dim)
            Phi = torch.exp(1j * x_ * ratio).diag()
            return Phi.kron(torch.eye(2))
        else:
            raise NotImplementedError
