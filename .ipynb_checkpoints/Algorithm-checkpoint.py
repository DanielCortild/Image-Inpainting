from tqdm import tqdm

from Image import Image

class Algorithm:
    """
    Solves problems of the type min f(x) + g(x) + h(Lx)
    Parameters:
        prox_f                Proximal Operator of f
        prox_g                Proximal Operator of g
        grad_h                Gradient of function h
        L                     Linear Operator L
        L_adj                 Adjoint of L
        Z0                    Initial guess for Z0
        Z1                    Initial guess for Z1
        alpha                 Initial value for alpha (Default: 1)
        alpha_static          Whether alpha is static or not (Default: True)
        lamb                  Value of lambda (Default: 1)
        rho                   Value of rho (Default: 1)
    Public Methods:
        run                   Runs the algorithm 
    Protected Methods:
    Private Methods:
        iterate               Runs a single iteration of the algorithm
    """
    
    def __init__(self, 
                 prox_f: callable,
                 prox_g: callable,
                 grad_h: callable,
                 L: callable,
                 L_adj: callable,
                 Z0: list,
                 Z1: list,
                 alpha: float = 0, 
                 alpha_static: bool = True,
                 lamb: float = 1, 
                 rho: float = 1) -> None:
        
        # Store the given functions
        self.__prox_f = lambda Z: prox_f(Z, rho)
        self.__prox_g = lambda Z: prox_g(Z, rho)
        self.__grad_h = grad_h
        self.__L = L
        self.__L_adj = L_adj
        
        # Create parameters for the iterations
        if alpha_static: self.get_alpha = lambda k: alpha
        else: self.get_alpha = lambda k: (1-1/k) * alpha
        self.lamb = lamb
        self.rho = rho
                
        # Set initial values
        self.Z0 = Z0
        self.Z1 = Z1

    def __iterate(self, Z0: list, Z1: list, k: int) -> tuple:
        """ @private
        Perform the iterations according to Algorithm 2
        """
        # Inertial Step
        alpha = self.get_alpha(k)
        U = Z1 + alpha * (Z1 - Z0)
        
        # Krasnoselskii-Mann Step
        XB = self.__prox_g(U)
        XA = self.__prox_f(2 * XB - U - self.rho * self.__L_adj(self.__grad_h(self.__L(XB))))
        Z2 = U + self.lamb * (XA - XB)
        
        # Bregman Iteration
        # Yet to be implemented
        
        return Z1, Z2
    
    def run(self, iterations: int) -> list:
        """ @public
        Run the algorithm given the number of iterations and the iterator
        """
        Z0, Z1 = self.Z0, self.Z1
        for i in tqdm(range(iterations)):
            Z0, Z1 = self.__iterate(Z0, Z1, i + 1)
        return Z1