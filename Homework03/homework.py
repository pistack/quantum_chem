from typing import Callable, Union, Tuple
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_LA


def make_T_1d(x_min: float, x_max: float,
              grid_space: float) -> sparse.csc.csc_matrix:

    '''
    Generates kinetic operator with CSC format

    Args:
    x_min: minimum value of domain
    x_max: maximum value of domain
    grid_space: grid spacing

    Returns:
    Kinetic operator matrix with CSC format
    '''

    num = np.arange(x_min, x_max, grid_space).shape[0]
    row_T = np.zeros(0, dtype=int)
    col_T = np.zeros(0, dtype=int)
    data_T = np.zeros(0, dtype=int)

    # (0, 0)
    row_T = np.concatenate((row_T, np.zeros(2, dtype=int)))
    col_T = np.concatenate((col_T, np.array([0, 1], dtype=int)))
    data_T = np.concatenate((data_T, np.array([-2, 1])))

    for i in range(1, num-1):
        # (i, 0)
        row_T = np.concatenate((row_T, i*np.ones(3, dtype=int)))
        tmp = np.array([(i-1), i, (i+1)], dtype=int)
        col_T = np.concatenate((col_T, tmp))
        data_T = np.concatenate((data_T, np.array([1, -2, 1])))

    # (nnum-1, 0)
    row_T = np.concatenate((row_T, num-1*np.ones(2, dtype=int)))
    col_T = np.concatenate((col_T, np.array([num-2,
                                             num-1],
                                            dtype=int)))
    data_T = np.concatenate((data_T, np.array([1, -2])))

    data_T = -0.5/(grid_space**2)*data_T
    T = sparse.csc_matrix((data_T, (row_T, col_T)), shape=(num, num))

    return T


def morse(x: Union[float, np.ndarray],
          k: float,  D: float) -> np.ndarray:
    '''
    Morse potential

    Args:
    x: distance from equivalent position
    k: force constant
    D: well depth

    Returns:
    array of Morse potential
    '''

    a = np.sqrt(k/(2*D))

    return D*(1-np.exp(-a*x))**2


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


def make_V(f: Callable, x_min: float, x_max: float,
           grid_space: float,
           args: Tuple = None) -> Tuple[np.ndarray, sparse.csc.csc_matrix]:

    '''
    Generates potential operator of given potential

    Args:
    f: potential
    x_min: minimum value of domain
    x_max: maximum value of domain
    grid_space: gird spacing
    args: additional argument for function f

    Returns:
      1. Diagonal elements for potential operator V
      2. Potential operator V (CSC format)
    '''

    x = np.arange(x_min, x_max, grid_space)
    num = x.shape[0]
    row = np.arange(0, num, 1, dtype=int)

    if args is None:
        diag_V = f(x)
    else:
        diag_V = f(x, *args)

    V = sparse.csc_matrix((diag_V, (row, row)), shape=(num, num))

    return diag_V, V


if __name__ == '__main__':

    # Define domain
    x_min = -10
    x_max = 10
    grid_space = 0.01

    # Define force constant
    k = 40.0

    # Define morse potential wall depth
    D = 100.0

    # Evaluate kinetic operator T
    T = make_T_1d(x_min, x_max, grid_space)

    # Evaluate potential operator of Morse potential
    diag_morse, V_morse = \
        make_V(morse, x_min, x_max, grid_space, args=(k, D))

    # Evaluate potential operator of Harmonic potential
    diag_harmonic, V_harmonic = \
        make_V(harmonic, x_min, x_max, grid_space, args=(k, ))

    # To faster arpack convergence, I use shift invert with sigma 0
    # instead of SM

    # Directly solve with Morse potential
    # With first 5 eigen value
    eigval_morse, eigvec_morse = \
        sparse_LA.eigsh(T+V_morse, k=5, sigma=0, which='LM')

    # First solve with Harmonic potential
    # For 2nd correction term, evaluate 20 eigen value and eigen vector
    eigval_harmonic, eigvec_harmonic = \
        sparse_LA.eigsh(T+V_harmonic, k=20, sigma=0, which='LM')

    # Calculate the correction term
    corr_pot = (diag_morse - diag_harmonic)
    corr_mat = (corr_pot * eigvec_harmonic.T) @ eigvec_harmonic

    # 1st correction term
    corr_1st = corr_mat.diagonal().copy()

    # Remove the diagonal elements in corr_mat for further correction
    corr_mat[np.eye(corr_mat.shape[0], dtype=bool)] = 0

    # Scale correction matrix
    tmp = eigval_harmonic.reshape(eigval_harmonic.shape[0], 1)
    e_diff = tmp @ np.ones_like(tmp).T - eigval_harmonic
    e_diff = np.eye(e_diff.shape[0]) + e_diff
    scaled_corr_mat = 1/e_diff*corr_mat

    # 2nd correction term
    corr_2nd = np.einsum('in,ni->n', corr_mat, scaled_corr_mat)

    print(f'Grid spacing: {grid_space}')
    print(f'Domain: [{x_min}, {x_max}]')
    print(f'Force constant: {k}')
    print(f'Wall depth: {D}')
    print('Eigenvalues')
    print('Direct    \t  Harmonic   \t     +1st \t     +2nd')
    for i in range(5):
        string = f'{eigval_morse[i]:.5f} \t {eigval_harmonic[i]: .5f} \t'
        string = string + f'{eigval_harmonic[i]+corr_1st[i]: .5f} \t'
        string = string + f'{eigval_harmonic[i]+corr_1st[i]+corr_2nd[i]: .5f}'
        print(string)
