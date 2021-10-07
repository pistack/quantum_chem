import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_LA
import matplotlib.pyplot as plt

# define grid spacing
grid_space = 0.01
x = np.arange(0-grid_space, 1+grid_space, grid_space)
nx = x.shape[0]
y = np.arange(0-grid_space, 1+grid_space, grid_space)
ny = y.shape[0]

# Define Kinetic Operator T
row_T = np.zeros(0, dtype=int)
col_T = np.zeros(0, dtype=int)
data_T = np.zeros(0)

for i in range(1, nx-1):
    for j in range(1, ny-1):
        row_T = np.concatenate((row_T, (i*ny+j)*np.ones(5, dtype=int)))
        tmp = np.array([(i-1)*ny+j, i*ny+(j-1),
                        i*ny+j, (i+1)*ny+j, i*ny+(j+1)], dtype=int)
        col_T = np.concatenate((col_T, tmp))
        data_T = np.concatenate((data_T, np.array([1, 1, -4, 1, 1])))

# Now deal with edge case
# missing two (0, ny-1), (nx-1, 0)
# (0, 0), (nx-1, ny-1), 
# (0, j) ( 1 < j < ny-2)
# (i, 0) ( 1 < i < nx-2)
# (i, ny-1) ( 1 < i < nx-2)
# (nx-1, j) ( 1 < j < ny-2)

# (0, 0)
row_T = np.concatenate((row_T, np.zeros(3, dtype=int)))
col_T = np.concatenate((col_T, np.array([0, ny, 1], dtype=int)))
data_T = np.concatenate((data_T, np.array([-4, 1, 1])))


# (0, ny-1)
row_T = np.concatenate((row_T, (ny-1)*np.ones(3, dtype=int)))
col_T = np.concatenate((col_T, np.array([ny-2, ny-1, 2*ny-1], dtype=int)))
data_T = np.concatenate((data_T, np.array([1, -4, 1])))

# (nx-1, 0)
row_T = np.concatenate((row_T, ((nx-1)*ny)*np.ones(3, dtype=int)))
col_T = np.concatenate((col_T, np.array([(nx-2)*ny,
                                         (nx-1)*ny, (nx-1)*ny+1],
                                        dtype=int)))
data_T = np.concatenate((data_T, np.array([1, -4, 1])))


# (nx-1, ny-1)
row_T = np.concatenate((row_T, (nx*ny-1)*np.ones(3, dtype=int)))
col_T = np.concatenate((col_T, np.array([(nx-2)*ny+ny-1,
                                         (nx-1)*ny+ny-2, (nx-1)*ny+ny-1],
                                        dtype=int)))
data_T = np.concatenate((data_T, np.array([1, 1, -4])))


# (0, j), (nx-1, j)
for j in range(1, ny-1):
    # (0, j)
    row_T = np.concatenate((row_T, j*np.ones(4, dtype=int)))
    tmp = np.array([j-1, j, ny+j, j+1], dtype=int)
    col_T = np.concatenate((col_T, tmp))
    data_T = np.concatenate((data_T, np.array([1, -4, 1, 1])))
    # (nx-1, j)
    row_T = np.concatenate((row_T, ((nx-1)*ny+j)*np.ones(4, dtype=int)))
    tmp = np.array([(nx-2)*ny+j, (nx-1)*ny+(j-1),
                    (nx-1)*ny+j, (nx-1)*ny+(j+1)], dtype=int)
    col_T = np.concatenate((col_T, tmp))
    data_T = np.concatenate((data_T, np.array([1, 1, -4, 1])))

# (i, 0), (i, ny-1)
for i in range(1, nx-1):
    # (i, 0)
    row_T = np.concatenate((row_T, i*ny*np.ones(4, dtype=int)))
    tmp = np.array([(i-1)*ny, i*ny, (i+1)*ny, i*ny+1], dtype=int)
    col_T = np.concatenate((col_T, tmp))
    data_T = np.concatenate((data_T, np.array([1, -4, 1, 1])))
    # (i, ny-1)
    row_T = np.concatenate((row_T, (i*ny+ny-1)*np.ones(4, dtype=int)))
    tmp = np.array([(i-1)*ny+ny-1, i*ny+ny-2,
                    i*ny+ny-1, (i+1)*ny+ny-1], dtype=int)
    col_T = np.concatenate((col_T, tmp))
    data_T = np.concatenate((data_T, np.array([1, 1, -4, 1])))

data_T = -0.5/(grid_space**2)*data_T
T = sparse.csc_matrix((data_T, (row_T, col_T)), shape=(nx*ny, nx*ny))

# Define potential operator V
row_V = np.array([0, nx*ny-1], dtype=int)
for j in range(1, ny-1):
    row_V = np.concatenate((row_V, np.array([j, (nx-1)*ny+j], dtype=int)))
for i in range(1, nx-1):
    row_V = np.concatenate((row_V, np.array([i*ny, i*ny+ny-1], dtype=int)))

V_MAX = 10**5
data_V = V_MAX*np.ones(row_V.shape[0], dtype=float)

V = sparse.csc_matrix((data_V, (row_V, row_V)), shape=(nx*ny, nx*ny))


# Because it gets two unphysical eigenvalue 0,
# we need to get 7 eigenvalues instead of 5
eigval, psi = sparse_LA.eigsh(T+V, k=7, which='SM')
eigval = eigval.real

# remove two eigenvalue and eigenvectors corresponding to zero
mask = np.abs(eigval) >= 1e-5
eigval = eigval[mask]
psi = psi[:, mask]
psi = psi.real

print('Particle in the 2D box')
print('Calculation Condtion: ')
print('Unit: atomic unit')
print('Length of box: 1 bohr')
print('Mass of particle: mass of electron')
print(f'Domain: [{0-grid_space},{1+grid_space}]^2')
print(f'Grid spacing: {grid_space}')
print('Potential V')
print('V(x,y) = 0 if (x,y) in [0,1]^2')
print(f'V(x,y) = {V_MAX:.1e}, otherwise')

# Note for particle in the box problem
# E_n = pi**2*hbar**2/(2*m*L**2)*n**2
# Since we set hbar=m=L=1, E_n = pi**2/2*n**2

for i in range(5):
    print(f'{i+1}th eigenvalue is {eigval[i]:.3f}')
    print(f'{i+1}th n_x**2+n_y**2 is {2*eigval[i]/np.pi**2: .0f}') 

X, Y = np.meshgrid(x, y)

for i in range(5):
    plt.figure(i+1)
    plt.title(f'eigval: {eigval[i]:.3f}')
    ax3d = plt.axes(projection="3d")
    ax3d.plot_surface(X, Y, -psi[:, i].reshape((nx, ny)), cmap='coolwarm',
                      linewidth=1)
plt.show()

        



