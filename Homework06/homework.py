from pyscf import gto, scf

zeta_list = ['d', 't', 'q', '5']
atomic_energy_list = []
molecule_energy_list = []

for zeta in zeta_list:
    # H atom
    h = gto.M(
        atom = 'H 0 0 0',
        spin = 1,
        charge = 0,
        verbose = 0,
        basis = f'cc-pv{zeta}z'
    )
    # H2 molecule
    h2 = gto.M(
        atom = 'H 0 0 0; H 0.74 0 0',
        spin = 0,
        charge = 0,
        verbose = 0,
        basis = f'cc-pv{zeta}z'
    )
    HF_atom = scf.UHF(h)
    e_atom = HF_atom.kernel()
    HF_molecule = scf.UHF(h2)
    e_mol = HF_molecule.kernel()
    atomic_energy_list.append(e_atom)
    molecule_energy_list.append(e_mol)

unit_convert = 627.50960803 # Hartree to kcal/mol
print("H2 molecule bonding energy")
print("Basis set \t Bonding energy (kcal/mol)")
for i in range(4):
    e_bond = molecule_energy_list[i] - \
        2*atomic_energy_list[i]
    e_bond = unit_convert*e_bond
    print(f"cc-pv{zeta_list[i]}z \t {e_bond:.6f}")