import numpy as np
import matplotlib.pyplot as plt # plotting density
from pyscf import gto, scf
from pyscf import lo # localized orbital

buta = gto.M(atom='''C     0.000000     0.000000     0.000000;
C     1.111831     0.743014     0.000000;
H     2.098555     0.277214     0.000000;
H     1.068887     1.831228     0.000000;
H    -0.978124     0.487592     0.000000;
C     0.000000    -1.462597     0.000000;
C    -1.111831    -2.205610     0.000000;
H    -2.098556    -1.739811     0.000000;
H     0.978125    -1.950189     0.000000;
H    -1.068889    -3.293825     0.000000''',
basis='cc-pvdz',
spin=0,
charge=0,
verbose=0)
hf_buta = scf.UHF(buta)
hf_buta.kernel()
C = hf_buta.mo_coeff
occ = hf_buta.mo_occ
occ = occ.astype(dtype='bool')
# Calc localized orbital
C_ibo_alpha =lo.ibo.ibo(buta, C[0,:,occ[0,:]].T)
C_ibo_beta = lo.ibo.ibo(buta, C[1,:,occ[1,:]].T)
# Calc density matrix using localized orbital
D_ibo_alpha = np.einsum('ik,jk-> ij', 
C_ibo_alpha, np.conj(C_ibo_alpha))
D_ibo_beta = np.einsum('ik,jk-> ij', 
C_ibo_beta, np.conj(C_ibo_beta))
D_ibo = D_ibo_alpha + D_ibo_beta
print(D_ibo.shape)
# Calc density matrix using normal orbital
D = hf_buta.make_rdm1()
D = D[0]+D[1]
print("Molecule: Butadiene")
print("Calculation Method: Hartree Fock")
print("Basis set: cc-pvdz")
print("Is two density matrix are same?")
print(np.allclose(D_ibo, D, rtol=1e-8, atol=1e-8))