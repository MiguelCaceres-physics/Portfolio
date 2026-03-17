import numpy as np 
L=8
lamda=5
alpha=2
x0=0.5
N=2000
#Malla
x=np.linspace(-L,L,N+1)
h=x[1]-x[0]
xi=x[1:-1]
#Potencial
V=0.5*xi**2+lamda*np.exp(-alpha*(xi-x0)**2)
#Laplaciano en 1D 2º orden (dirichlet) en interior
main=np.full(N-1,-2.0)/h**2
off=np.full(N-2,1.0)/h**2
#Hamiltoiano
H=np.diag(-0.5*main)+np.diag(-0.5*off,1)+np.diag(-0.5*off,-1)+np.diag(V)
#autovalores y autovectores
E,U=np.linalg.eigh(H)
k=6
E=E[:k]
U=U[:,:k]
#Normalizacion L2
norm=np.sqrt(np.sum(U**2,axis=0)*h)
U=U/norm
# Observables
x_mean=(U**2*xi[:,None]).sum(axis=0)*h
x2_mean=(U**2*xi[:,None]**2).sum(axis=0)*h
print("E (primeros 6 valores):",E)
print("<x>:",x_mean)
print("<x^2>:",x2_mean)
#numero de nodos
def nodos(u):
    return np.sum(u[1:]*u[:-1]<0)
print("Nodos (primeros 6 estados):" ,[nodos(U[:,i]) for i in range(k)])

