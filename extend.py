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
    # r parameter of damping
    r = np.sqrt(gamma ** 2 / 4 - g ** 2)
    # setting operator
    self.G = torch.tensor([
        [gamma / (2 * r) * np.sinh(r) + np.cosh(r) , - 1j * g * np.sinh(r) / r],
        [- 1j * g * np.sinh(r) / r, -gamma / (2 * r) * np.sinh(r) + np.cosh(r)]
        ])
    
  def set_G_inv(self, g : float , gamma : float) -> None : 
    # r parameter of damping
    r = np.sqrt(gamma ** 2 / 4 - g ** 2)
    # setting operator
    self.G = torch.tensor([
        [- gamma / (2 * r) * np.sinh(r) + np.cosh(r) , 1j * g * np.sinh(r) / r],
        [ 1j * g * np.sinh(r) / r, gamma / (2 * r) * np.sinh(r) + np.cosh(r)]
        ])

  def set_Cup(self, thp : float) -> None :
    # setting operator
    self.Cup = torch.tensor([
        [np.cos(thp / 2) , - np.sin(thp / 2)],
        [np.sin(thp / 2) ,   np.cos(thp / 2)]
    ])

  def set_Sup(self, k : int) -> None :
    # setting operator
    self.Sup = torch.tensor([
        [np.exp(- 1j * 2 * np.pi / self.dim * k), 0],
        [0, 1]
    ])

  def set_Sdwn(self, k : int) -> None :
    # setting operator
    self.Sup = torch.tensor([
        [1, 0],
        [0, np.exp(1j * 2 * np.pi / self.dim * k)]
    ])

  def set_Cdwn(self, thp : float) -> None:
    # setting operator
    self.Cdwn = torch.tensor([
        [np.cos(thp / 2) , - np.sin(thp / 2)],
        [np.sin(thp / 2) ,   np.cos(thp / 2)]
    ])

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
    return self.Sup @ self.Cup @ self.G @ self.Sdwn @ self.Cdwn @ self.G_inv
  