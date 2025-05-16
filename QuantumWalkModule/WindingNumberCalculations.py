from torch import nn as nn
import torch
import torch.linalg as lin
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")

SX = torch.tensor([
    [0,1],
    [1,0]
])

SY = torch.tensor([
    [0,-1j],
    [1j,0]
])

SZ = torch.tensor([
    [1,0],
    [0,-1]
])

ID = torch.tensor([
    [1,0],
    [0,1]
])

class NonHermitian:
    def __init__(self, dim = 10):
        # number of nodes
        self.dim = dim
        # operators
        self.G = None
        self.G_inv = None
        self.Cup = None
        self.Sup = None
        self.Cdwn = None
        self.Sdwn = None

    def check_all(self):
        # check wether all operators, but step  are initialized
        ops = [self.G, self.G_inv, self.Cup, self.Cdwn]
        for op in ops:
            if op == None:
                return False
        return True

    def set_G(self, g : float , gamma : float) -> None : 
        # setting operator
        if g == 0 and gamma == 0:
            self.G = torch.eye(2).to(torch.complex64)
        else:
            self.G = torch.tensor(expm(-1j * g * SX + gamma / 2 * SZ)).to(torch.complex64)
        
    def set_G_inv(self, g : float , gamma : float) -> None : 
        # setting operator
        if g == 0 and gamma == 0:
            self.G_inv = torch.eye(2).to(torch.complex64)
        else:
            self.G_inv = torch.tensor(expm(1j * g * SX - gamma / 2 * SZ)).to(torch.complex64)

    def set_Cup(self, thp : float) -> None :
        # setting operator
        self.Cup = torch.tensor([
            [np.cos(thp / 2) , - np.sin(thp / 2)],
            [np.sin(thp / 2) ,   np.cos(thp / 2)]
        ]).to(torch.complex64)

    def set_Sup(self, k : int) -> None :
        # setting operator
        self.Sup = torch.tensor([
            [np.exp(- 1j * 2 * np.pi / self.dim * k), 0],
            [0                                      , 1]
        ]).to(torch.complex64)

    def set_Sdwn(self, k : int) -> None :
        # setting operator
        self.Sdwn = torch.tensor([
            [1,                                     0],
            [0, np.exp(1j * 2 * np.pi / self.dim * k)]
        ]).to(torch.complex64)

    def set_Cdwn(self, thp : float) -> None:
        # setting operator
        self.Cdwn = torch.tensor([
            [np.cos(thp / 2) , - np.sin(thp / 2)],
            [np.sin(thp / 2) ,   np.cos(thp / 2)]
        ]).to(torch.complex64)

    def set_operators_2(self, g, gamma,thm, thp) -> None:
        # set operators
        self.set_G(g, gamma)
        self.set_G_inv(g, gamma)
        self.set_Cup(thp)
        self.set_Cdwn(thm)

    def set_step_operators(self,k):
        # set operators
        self.set_Sup(k)
        self.set_Sdwn(k)

    def set_operators(self, g, gamma, k, thm, thp ) -> torch.tensor:
        # set operators
        self.set_G(g, gamma)
        self.set_G_inv(g, gamma)
        self.set_Sup(k)
        self.set_Sdwn(k)
        self.set_Cup(thp)
        self.set_Cdwn(thm)

        # return U
        return self.get_U()

    def get_U(self):
        return torch.matmul(self.Sup, torch.matmul(self.Cup , torch.matmul(self.G ,torch.matmul(self.Sdwn ,torch.matmul( self.Cdwn , self.G_inv)))))

class BerryPhaseCalculation:
    def __init__(self, dim = 100):
        self.dim = dim
        # initialize by setting non hermitian hamiltonian
        self.system = NonHermitian(dim)

    def prepare_operators(self, g, gamma, thm, thp):
    # set parameter values for operators
        self.system.set_operators_2(g, gamma, thm, thp)

    def calc_W(self, mode = "upper"):
        assert mode in ["upper","lower"], "invalid mode type"
        assert self.system.check_all() , "you must initialize operators first"

        # main loop: for each k do : -> calculate eigenvectors and stack them 
        for k in range(self.dim):
            self.system.set_step_operators(k)
            U = self.system.get_U()
            E, V = lin.eig(U)
            E = np.angle(E)
            V  = V / np.sqrt(torch.diag(V.T.conj() @ V)).reshape(1,-1)
            if k == 0:
                vecs = V[:,E >= 0].reshape(2,-1)
            else :
                vecs = torch.concat((vecs,V[:,E >= 0].reshape(2,-1)), axis = 1)

        # shift the matrix along 0 axis and multiply to form a chain
        chain = vecs.T.conj().roll(1,0) @ vecs

        # return Winding number
        prod = torch.prod(chain)
        return np.abs(np.angle(prod)) / np.pi

if __name__ == "__main__":
    ths = np.linspace(- 2 * np.pi, 2 * np.pi , 10)

    Winding_num = np.zeros((len(ths),len(ths)), dtype = complex)

    model = BerryPhaseCalculation()

    g = 0
    gamma = 0

    for i, thp in enumerate(ths):
        for j, thm in enumerate(ths):
            model.prepare_operators(g, gamma, thp, thm)
            el = model.calc_W().item()
            Winding_num[i,j] = el

    fig = plt.figure(figsize = (10,5))

    ax = fig.add_subplot(1,2,1)
    ax.imshow(Winding_num.real)
    ax.grid(False)

    ax = fig.add_subplot(1,2,2)
    ax.imshow(Winding_num.imag)
    ax.grid(False)

    plt.show()