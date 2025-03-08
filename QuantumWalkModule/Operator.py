import torch

class Op:
    sx = torch.tensor([
        [0,1],
        [1,0]
    ], dtype = torch.complex64)

    sy = torch.tensor([
        [0 ,-1j],
        [1j,  0]
    ], dtype = torch.complex64)

    sz = torch.tensor([
        [1, 0],
        [0,-1]
    ], dtype = torch.complex64)

    sp = torch.tensor([
        [0,1],
        [0,0]
    ], dtype = torch.complex64)

    sm = sp.conj().T

    s_r = torch.tensor([
        [1,0],
        [0,0]
    ], dtype = torch.complex64)

    s_l = torch.tensor([
        [0,0],
        [0,1]
    ], dtype = torch.complex64)

