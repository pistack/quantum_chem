from pyscf import gto
import numpy as np
from scipy.linalg import eigh
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_LA
import matplotlib.pyplot as plt

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
    T = sparse.diags(np.array([1, -2, 1]), np.array([-1, 0, 1]),
                     shape=(num, num))

    T = -0.5/(grid_space**2)*T

    return T

def get_J(D, ERI):
    J = np.einsum('rs,pqrs -> pq', D, ERI)
    return J

def rhf_scf(mol, init_guess=None):
    '''
    Do RHF self consistent field
    Hartree fock process

    Args:
    mol: pyscf molecule
    init_guess: initial density matrix (alpha only)

    Returns:
    total energy of molecule,
    orbital energy,
    orbital coefficients,
    number of occupied orbital
    '''
    converged = False
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    S = mol.intor('int1e_ovlp')
    ERI = mol.intor('int2e')
    Nocc = mol.nelectron//2
    # get initial guess used to contruct 
    # fock matrix
    e, C = eigh(T+V, S)
    # initial energy 
    # for non interacting system total energy is same as
    # sum of indivisual eigen value
    E = 2*np.sum(e[:Nocc])
    D = np.einsum('pk,qk->pq', C[:, :Nocc], np.conj(C[:, :Nocc]))
    if init_guess is not None:
        D = init_guess
    # generate J columb matrix
    J = np.einsum('rs,pqrs -> pq', D, ERI)
    # generate K exchange matrix
    K = np.einsum('rq,pqrs -> ps', D, ERI)
    # Now construct first fock matrix
    F = T+V+2*J-K
    # update guess
    ep, Cp = eigh(F, S)
    Dp = np.einsum('ik,jk->ij', Cp[:, :Nocc], np.conj(Cp[:, :Nocc]))
    # update energy
    Ep = np.einsum('ij,ij',T+V+F, Dp)

    #start scf cycle
    while(not converged):
        D = Dp
        E = Ep
        # generate J columb matrix
        J = np.einsum('rs,pqrs -> pq', D, ERI)
        # generate K exchange matrix
        K = np.einsum('rq,pqrs -> ps', D, ERI)
        # Now construct first fock matrix
        F = T+V+2*J-K
        # update guess
        ep, Cp = eigh(F, S)
        Dp = np.einsum('ik,jk->ij', Cp[:, :Nocc], np.conj(Cp[:, :Nocc]))
        # update energy
        Ep = np.einsum('ij,ij',T+V+F, Dp)
        # check convergence
        converged = (np.abs(E-Ep) < 1e-9)
    
    # Nuclear Nuclear repulsion energy
    EN = mol.energy_nuc()
    return Ep+EN, ep, Cp, Nocc

def uhf_scf(mol,init_guess=None):
    '''
    Do UHF self consistent field
    Hartree fock process

    Args:
    mol: pyscf molecule
    init_guess: initial density matrix
    init_guess[0]: init alpha density matrix
    init_guess[1]: init beta density matrix

    Returns:
    total energy of molecule,
    orbital energy,
    orbital coefficients,
    number of occupied orbital
    '''
    converged = False
    T = mol.intor('int1e_kin')
    V = mol.intor('int1e_nuc')
    S = mol.intor('int1e_ovlp')
    ERI = mol.intor('int2e')
    Nocc_alpha = (mol.nelectron+mol.spin)//2
    Nocc_beta = (mol.nelectron-mol.spin)//2
    # get initial guess used to contruct 
    # fock matrix
    e, C = eigh(T+V, S)
    # initial energy 
    # for non interacting system total energy is same as
    # sum of indivisual eigen value
    E = np.sum(e[:Nocc_alpha])+np.sum(e[:Nocc_beta])
    D_alpha = np.einsum('pk,qk->pq', C[:, :Nocc_alpha], 
    np.conj(C[:, :Nocc_alpha]))
    D_beta = np.einsum('pk,qk->pq', C[:, :Nocc_beta], 
    np.conj(C[:, :Nocc_beta]))
    if init_guess is not None:
        D_alpha = init_guess[0]
        D_beta = init_guess[1]
    # generate J columb matrix
    J_alpha = np.einsum('rs,pqrs -> pq', D_alpha, ERI)
    J_beta = np.einsum('rs,pqrs -> pq', D_beta, ERI)
    # generate K exchange matrix
    K_alpha = np.einsum('rq,pqrs -> ps', D_alpha, ERI)
    K_beta = np.einsum('rq,pqrs -> ps', D_beta, ERI)
    # Now construct first fock matrix
    F_alpha = T+V+J_alpha+J_beta-K_alpha
    F_beta = T+V+J_alpha+J_beta-K_beta
    # update guess
    ep_alpha, Cp_alpha = eigh(F_alpha, S)
    ep_beta, Cp_beta = eigh(F_beta, S)
    Dp_alpha = np.einsum('ik,jk->ij', 
    Cp_alpha[:, :Nocc_alpha], np.conj(Cp_alpha[:, :Nocc_alpha]))
    Dp_beta = np.einsum('ik,jk->ij', 
    Cp_beta[:, :Nocc_beta], np.conj(Cp_beta[:, :Nocc_beta]))
    # update energy
    Ep = 0.5*(np.einsum('ij,ij',T+V+F_alpha, Dp_alpha) + 
    np.einsum('ij,ij',T+V+F_beta, Dp_beta))

    #start scf cycle
    while(not converged):
        D_alpha = Dp_alpha
        D_beta = Dp_beta
        E = Ep
        # generate J columb matrix
        J_alpha = np.einsum('rs,pqrs -> pq', D_alpha, ERI)
        J_beta = np.einsum('rs,pqrs -> pq', D_beta, ERI)
        # generate K exchange matrix
        K_alpha = np.einsum('rq,pqrs -> ps', D_alpha, ERI)
        K_beta = np.einsum('rq,pqrs -> ps', D_beta, ERI)
        # Now construct first fock matrix
        F_alpha = T+V+J_alpha+J_beta-K_alpha
        F_beta = T+V+J_alpha+J_beta-K_beta
        # update guess
        ep_alpha, Cp_alpha = eigh(F_alpha, S)
        ep_beta, Cp_beta = eigh(F_beta, S)
        Dp_alpha = np.einsum('ik,jk->ij', 
        Cp_alpha[:, :Nocc_alpha], np.conj(Cp_alpha[:, :Nocc_alpha]))
        Dp_beta = np.einsum('ik,jk->ij', 
        Cp_beta[:, :Nocc_beta], np.conj(Cp_beta[:, :Nocc_beta]))
        # update energy
        Ep = 0.5*(np.einsum('ij,ij',T+V+F_alpha, Dp_alpha) + 
        np.einsum('ij,ij',T+V+F_beta, Dp_beta))
        # check convergence
        converged = (np.abs(E-Ep) < 1e-9)

    
    # Nuclear Nuclear repulsion energy
    EN = mol.energy_nuc()
    ep = np.zeros((2,ep_alpha.shape[0]))
    Cp = np.zeros((2,Cp_alpha.shape[0], Cp_alpha.shape[1]))
    Nocc = np.zeros(2)
    ep[0,:] = ep_alpha
    ep[1,:] = ep_beta
    Cp[0,:,:] = Cp_alpha
    Cp[1,:,:] = Cp_beta
    Nocc[0] = Nocc_alpha
    Nocc[1] = Nocc_beta
    return Ep+EN, ep, Cp, Nocc

if __name__ == "__main__":
    # Draw potential energy surface of LiH linear molecule
    dist = np.arange(2, 6.1, 0.1) # (unit bohr)
    ang_to_bohr = 1.8897259886
    au_to_invcm = 2.1947*10**5
    energies_rhf = np.zeros_like(dist)
    energies_uhf = np.zeros_like(dist)
    for i,d in enumerate(dist):
        LiH = gto.M(atom=[['Li', 0, 0, 0], ['H', 
        d/ang_to_bohr, 0, 0]], 
        charge=0, spin=0, basis='6-31g', verbose=0)
        energies_rhf[i], orb_energy, mo_coeff, num_occ \
            = rhf_scf(LiH)
        energies_uhf[i], orb_energy, mo_coeff, num_occ \
            = uhf_scf(LiH)
    
    # Evaluate nuclear eigenvalue via finite difference method
    T = make_T_1d(2, 6.1, 0.1)
    mu = 7./8.*1822.89
    T = T/mu
    V_rhf = sparse.diags(energies_rhf-np.min(energies_rhf), 
    shape=T.shape)
    V_uhf = sparse.diags(energies_uhf-np.min(energies_uhf), 
    shape=T.shape)
    eigval_rhf, _ = sparse_LA.eigsh(T+V_rhf, k=10, which='SM')
    eigval_uhf, _ = sparse_LA.eigsh(T+V_uhf, k=10, which='SM')
    fdm_rhf = au_to_invcm*eigval_rhf
    fdm_uhf = au_to_invcm*eigval_uhf
    # Harmonic approximation
    # E_v = (v+1/2)hbar*w
    # E_v = (v+1/2)hbar*sqrt(k/mu)
    # k = second derivative of electronic energy
    # unit a.u./bohr**2
    # second derivative can be approximated to
    # f''(x) ~ (f(x+h)+f(x-h)-2f(x))/h**2
    min_idx_rhf = np.argmin(energies_rhf)
    min_idx_uhf = np.argmin(energies_uhf) 
    k_rhf = (energies_rhf[min_idx_rhf-1]+
    energies_rhf[min_idx_rhf+1]-2*energies_rhf[min_idx_rhf])/0.01
    k_uhf = (energies_rhf[min_idx_uhf-1]+
    energies_rhf[min_idx_uhf+1]-2*energies_rhf[min_idx_uhf])/0.01
    wave_rhf = np.sqrt(k_rhf/mu)*au_to_invcm
    wave_uhf = np.sqrt(k_uhf/mu)*au_to_invcm

    print('Vibrational energies in cm^{-1} unit')
    print('v \t RHF (Harmonic) \t RHF (FDM) \t UHF (Harmonic) \t UHF (FDM)')
    for i in range(10):
        print(f'{i} \t {(i+1/2)*wave_rhf:.2f} \t {fdm_rhf[i]:.2f} \t {(i+1/2)*wave_uhf:.2f} \t {fdm_uhf[i]:.2f}')
    
    plt.plot(dist, energies_rhf, label='RHF')
    plt.plot(dist, energies_uhf, label='UHF')
    plt.plot(dist,0.5*k_rhf*(dist-dist[min_idx_rhf])**2
    +energies_rhf[min_idx_rhf], linestyle='dashed', label='RHF (Harmonic)')
    plt.plot(dist,0.5*k_uhf*(dist-dist[min_idx_uhf])**2
    +energies_uhf[min_idx_uhf], linestyle='dashed', label='UHF (Harmonic)')
    plt.xlabel('R (bohr)')
    plt.xlim(2, 6)
    plt.ylim(-7.99,-7.899)
    plt.legend()
    plt.show()
