import numpy as np
from pyscf import gto, scf

atom_lst = ['H', 'He',
'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']

spin_lst = [1, 0,
1, 0, 1, 2, 3, 2, 1, 0,
1, 0, 1, 2, 3, 2,1, 0]

homo_lst = []
lumo_lst = []

for a, s in zip(atom_lst, spin_lst):
    gto_atom = gto.M(
        atom=f'{a} 0 0 0',
        spin = s,
        charge = 0,
        verbose = 0,
        basis = 'cc-pv5z'
    )
    HF_atom = scf.UHF(gto_atom)
    HF_atom.kernel()
    e_mo = HF_atom.mo_energy
    occ = HF_atom.mo_occ
    occ = occ.astype(dtype='bool')
    # homo alpha orbital
    homo_lst.append(np.max(e_mo[0, occ[0, :]]))
    # lumo alpha orbital
    lumo_lst.append(np.min(e_mo[0, ~occ[0, :]]))
    # due to spin selection rule
    # homo alpha to lumo alpha transition is favored

print('Calculation Method: HF')
print('Reference: UHF')
print('Atom \t HOMO \t LUMO')
for a, homo, lumo in zip(atom_lst, homo_lst, lumo_lst):
    print(f'{a} \t {homo:.4f} \t {lumo:.4f}')
