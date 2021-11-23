import numpy as np
import matplotlib.pyplot as plt # plotting density
from pyscf import gto, scf
from pyscf import dft # density calculation

grid_space = 0.01 # grid spacing
atom_lst = ['H', 'He', 'Be', 'Ne', 'Mg', 'Ar']
spin_lst = [1, 0, 0, 0, 0, 0]

r = np.arange(0, 10+grid_space, grid_space)
coords = np.zeros((r.shape[0], 3))
coords[:, 0] = r

plt.figure(1) # set figure for plotting
print(f'Atom \t N_elec \t <r>')

# HF calcuation
for a, s in zip(atom_lst, spin_lst):
    gto_atom  = gto.M(
        atom = f'{a} 0 0 0',
        spin = s,
        charge = 0,
        verbose = 0,
        basis = 'cc-pv5z'
    )
    hf_atom = scf.UHF(gto_atom)
    hf_atom.kernel()
    ao = dft.numint.eval_ao(gto_atom, coords)
    # alpha density
    rho_alpha = dft.numint.eval_rho2(gto_atom, ao, hf_atom.mo_coeff[0, :, :],
    hf_atom.mo_occ[0, :])
    # beta density
    rho_beta = dft.numint.eval_rho2(gto_atom, ao, hf_atom.mo_coeff[1, :, :],
    hf_atom.mo_occ[1, :])
    rho = rho_alpha + rho_beta
    # atom is spherical harmonic
    # radical density
    r_den = 4*np.pi*r**2*rho
    # Number of electron 
    N_elec = np.sum(r_den)*grid_space
    r_mean = np.sum(r_den*r)*grid_space/N_elec
    print(f'{a} \t {N_elec:.5f} \t {r_mean:.5f}')
    plt.plot(r, r_den, label=a)

plt.xlim(0, 5)
plt.ylim(0, 25)
plt.legend()
plt.show()