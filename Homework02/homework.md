# Additional Homework

Let 
$$
A = \begin{pmatrix}
3 & -1 \\
-5 & 7
\end{pmatrix}
$$
Consider characteristic polynomial of $A$
$$
p(A)(\lambda) = \begin{vmatrix}
\lambda -3 & 1 \\
5 & \lambda -7
\end{vmatrix}
$$

Then, $p(A)(\lambda) = \lambda^2 - 10\lambda + 16$
Eigenvalues of $A$ satisfy $p(A)(\lambda)=0$. So, eigenvalues are 
$\lambda=2,8.$

* Eigenvalue: $2$
  Eigenvector: $(1,1)^T$
  
* Eigenvalue: $8$
  Eigenvector: $(1,5)^T$

## Compare to numpy linalg eig

* Eigenvalue: $2$
  Eigenvector: $(-\sqrt(2), -\sqrt(2))^T$

* Eigenvalue: $8$
  Eigenvector: $(0.196, -0.981)^T$

Eigenvalues are exactly same and the eigenvectors are same up to scale factor.
