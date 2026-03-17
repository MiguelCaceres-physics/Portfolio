import numpy as np
import matplotlib.pyplot as plt
L=10
a=2
V1=10
V2=8
N=200
#variables
x=np.linspace(-L,L,N+1)
h=x[1]-x[0]
xi=x[1:-1]
#potencial
V=np.zeros_like(x)
V=np.where(np.abs(x)>a,0,
np.where(x<0,-V1,-V2))
#Laplaciano 2º orden (dirichlet) en interior
main=np.full(N-1,-2)/h**2
off=np.full(N-2,1)/h**2
#Hamiltoniano
H=np.diag(-0.5*main)+np.diag(-0.5*off,1)+np.diag(-0.5*off,-1)+np.diag(V[1:-1])
#autovalores y autovectores 
E,U=np.linalg.eigh(H)
k=6
E=E[:k]
U=U[:,:k]
#Normalizacion L2
norm = np.sqrt(np.sum(U**2, axis=0) * h)
U=U/norm
#Graficas
plt.figure()
for i in range(k):
    plt.plot(xi,U[:,i]+E[i],label=f"n={i}, E={E[i]:.3f}")
plt.plot(x,V,'k--',label='V(x)')
plt.xlabel('x')
plt.ylabel('E y ψ(x)')
plt.title('Estados ligados en pozo doble asimétrico')
plt.grid()
plt.show()
print("E (primeros 6 valores):",E)


