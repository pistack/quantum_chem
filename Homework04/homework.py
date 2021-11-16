from typing import Callable, Union, Tuple
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_LA
import matplotlib.pyplot as plt


def make_T_2d(p_min: Tuple[float, float], p_max: Tuple[float, float],
              grid_space: float) -> sparse.csc.csc_matrix:

    '''
    Generates kinetic operator with CSC format

    Args:
    p_min: tuple of minimum value of domain (x, y)
    p_max: tuple of maximum value of domain (x, y)
    grid_space: grid spacing 

    Returns:
    Kinetic operator matrix with CSC format
    '''

    n_x = np.arange(p_min[0], p_max[0]+grid_space, grid_space).shape[0]
    n_y = np.arange(p_min[1], p_max[1]+grid_space, grid_space).shape[0]
    T = sparse.diags(np.array([1, 1, -4, 1, 1]), np.array([-n_y, -1, 0, 1, n_y]),
                     shape=(n_x*n_y, n_x*n_y))

    T = -0.5/(grid_space**2)*T

    return T


def gauss(x: Union[float, np.ndarray],
          a: float, b: float) -> np.ndarray:
    '''
    Gaussian potential

    Args:
    x: distance between particle
    a: repulsion constant
    b: repulsion energy at x = 0

    Returns:
    array of gaussian repulsion constant
    '''

    return b*np.exp(-a*x**2)


def harmonic(x: Union[float, np.ndarray], k: float) -> np.ndarray:
    '''
    Harmonic potential

    Args:
    x: distance from equivalent position
    k: force constant

    Returns:
    array of Harmonic potential
    '''

    return 0.5*k*x**2


def make_V1(f: Callable, p_min: Tuple[float, float], 
           p_max: Tuple[float, float],
           grid_space: float,
           args: Tuple = None) -> sparse.csc.csc_matrix:

    '''
    Generates potential operator of given potential of
    type g(x,y) = f(x) + f(y) 

    Args:
    f: potential
    p_min: minimum value of domain ( both x and y direction)
    p_max: maximum value of domain ( both x and y direction)
    grid_space: gird spacing
    args: additional argument for function f

    Returns:
    Potential operator V (CSC format)
    '''

    x, y = np.meshgrid(np.arange(p_min[0], p_max[0]+grid_space, grid_space),
    np.arange(p_min[1], p_max[1]+grid_space, grid_space))

    if args is None:
        diag_V = f(x) + f(y)
    else:
        diag_V = f(x, *args) + f(y, *args)

    diag_V = diag_V.flatten()

    V = sparse.diags(diag_V, format="csc")

    return V

def make_V2(f: Callable, p_min: Tuple[float, float], 
           p_max: Tuple[float, float],
           grid_space: float,
           args: Tuple = None) -> sparse.csc.csc_matrix:

    '''
    Generates potential operator of given potential of
    type g(x,y) = f(x-y) 

    Args:
    f: potential
    p_min: minimum value of domain ( both x and y direction)
    p_max: maximum value of domain ( both x and y direction)
    grid_space: gird spacing
    args: additional argument for function f

    Returns:
    Potential operator V (CSC format)
    '''

    x, y = np.meshgrid(np.arange(p_min[0], p_max[0]+grid_space, grid_space),
    np.arange(p_min[1], p_max[1]+grid_space, grid_space))

    if args is None:
        diag_V = f(x-y)
    else:
        diag_V = f(x-y, *args)

    diag_V = diag_V.flatten()

    V = sparse.diags(diag_V, format="csc")

    return V

def cond_prob(psi: np.ndarray, x_index: int) -> np.ndarray:
    '''
    Evaluates conditional pdf of given psi at fixed x value
    Args:
    psi: two dimensional wave function
    x_index: index of x value at which find conditional pdf
    Returns:
    Conditional pdf
    '''
    cond_pdf = psi[x_index, :]
    cond_pdf = cond_pdf / np.sum(cond_pdf)

    return cond_pdf


if __name__ == '__main__':

    # Define domain
    p_min = (-2, -2)
    p_max = (2, 2)
    grid_space = 0.1

    # Define force constant
    k = 40

    # repulsion constant
    a = 8

    # repulsion energy at distance = 0
    b = 12

    # x index to evaluate conditional pdf
    x_idxs = [0, # x_min
    int(1/grid_space), # x_min + 1
    int(2/grid_space), # x_min + 2 
    int(3/grid_space), # x_min + 3 
    int(4/grid_space), # x_min + 4
    ]


    # Evaluate kinetic operator T
    T = make_T_2d(p_min, p_max, grid_space)

    # Evaluate potential for non interacting harmonic potential
    V_non_int = make_V1(harmonic, p_min, p_max, grid_space, args=(k,))

    # Evaluate potential for repulsive gaussian potential
    V_repul = make_V2(gauss, p_min, p_max, grid_space, args=(a,b,))

    # To faster arpack convergence, I use shift invert with sigma 0
    # instead of SM

    # Solve non interacting system
    eigval_non_int, eigvec_non_int = \
        sparse_LA.eigsh(T+V_non_int, k=5, sigma=0, which='LM')
    
    # Solve interacting system
    eigval_int, eigvec_int = \
        sparse_LA.eigsh(T+V_non_int+V_repul, k=5, sigma=0, which='LM')
    
    # Evaluate conditional pdf
    n_x = int((p_max[0]-p_min[0])/grid_space)+1
    n_y = int((p_max[1]-p_min[1])/grid_space)+1
    y = np.arange(p_min[1], p_max[1]+grid_space, grid_space)
    xs = np.arange(p_min[0], p_max[0]+grid_space, grid_space)[x_idxs]
    cond_pdf_non_int = np.zeros((n_y,len(x_idxs)))
    cond_pdf_int = np.zeros((n_y,len(x_idxs))) 
    for i in range(len(x_idxs)):
        cond_pdf_non_int[:, i] = \
            cond_prob(eigvec_non_int[:, 0].reshape((n_x, n_y)), 
            x_idxs[i])
        cond_pdf_int[:, i] = \
            cond_prob(eigvec_int[:, 0].reshape((n_x, n_y)), 
            x_idxs[i])

    print(f'Grid spacing: {grid_space}')
    print(f'Domain: [{p_min[0]}, {p_max[0]}]^2')
    print(f'Force constant: {k}')
    print(f'Parameter a: {a}')
    print(f'Parameter b: {b}')
    print('Eigenvalues')
    print('Num \t Non interacting   \t  Interacting')
    for i in range(5):
        string = f'{i+1} \t {eigval_non_int[i]:.5f} \t {eigval_int[i]: .5f} \t'
        print(string)

    plt.figure(1)
    plt.title('Non interacting system: conditional pdf')
    for i in range(len(x_idxs)):
        plt.plot(y, cond_pdf_non_int[:, i], label=f'x={y[x_idxs[i]]:.2f}')
    plt.legend()
    plt.figure(2)
    plt.title('Interacting system: conditional pdf')
    for i in range(len(x_idxs)):
        plt.plot(y, cond_pdf_int[:, i], label=f'x={y[x_idxs[i]]:.2f}')
    plt.legend()
    plt.show()
