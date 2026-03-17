import numpy as np
import matplotlib.pyplot as plt
N=4000
Rmax=100
r=np.linspace(0,Rmax,N+1)
ri=r[1:-1]
h=r[1]-r[0]
Z=1
#Portencial
for ell in [0,1,2]:
    V=-Z/r[1:-1]+ell*(ell+1)/(2*r[1:-1]**2)
    #Laplaciano 2º orden (dirichlet) en interior
    main=np.full(N-1,-2.0)/h**2
    off=np.full(N-2,1.0)/h**2
    #Hamiltoniano
    H=np.diag(-0.5*main)+np.diag(-0.5*off,-1)+np.diag(-0.5*off,1)+np.diag(V)
    #autovalores y autovectores
    E,U=np.linalg.eigh(H)
    k=6
    E=E[:k]
    U=U[:,:k]
    #Normalizacion L2
    norm=np.sqrt(np.sum(U**2,axis=0)*h)
    U=U/norm
    #Graficas
    plt.figure()
    for i in range(k):
        plt.plot(ri,U[:,i]+E[i],label=f"n={i}, E={E[i]:.3f}")
    plt.plot(ri,V,'k--',label='V(r)')
    plt.xlabel('r')
    plt.ylabel('E y ψ(r)')
    plt.ylim(min(E)-0.8,0.2)
    plt.title(f'Estados ligados del hidrógeno, l={ell}')
    plt.grid()
    plt.savefig(f"/home/marijuani/Documentos/Programacion/ESTUDIO_PROPIO/GRAFICAS/hidrogeno_l{ell}.png", dpi=200)
    plt.show()
    plt.close()
    print(f"E (primeros 6 valores, ell={ell}):",E)
