import numpy as np
import matplotlib.pyplot as plt

L = 10
N = 400
omega = 1.0

x = np.linspace(-L, L, N+1)
xi = x[1:-1]
h = x[1] - x[0]

# Potencial armónico
V = 0.5 * omega**2 * xi**2

# Operador cinético (diferencias finitas 2º orden)
main = np.full(N-1, -2.0) / h**2
off  = np.full(N-2,  1.0) / h**2

T = (-0.5) * (np.diag(main) + np.diag(off, 1) + np.diag(off, -1))

# Hamiltoniano total
H = T + np.diag(V)

# Autovalores y autovectores
E, U = np.linalg.eigh(H)

k = 6
E = E[:k]
U = U[:, :k]

# Normalización L2 discreta: sum |psi|^2 h = 1
norm = np.sqrt(np.sum(U**2, axis=0) * h)
U = U / norm

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(xi, V, 'k', label='Potencial armónico')

for i in range(k):
    plt.plot(xi, U[:, i] + E[i], 'b')
    plt.hlines(E[i], -L, L, colors='r', linestyles='dashed')

plt.title('Estados del oscilador armónico cuántico 1D (FD + autovalores)')
plt.xlabel('x')
plt.ylabel('Energía y funciones de onda (desplazadas)')
plt.ylim(-0.5, V.max() + 2)
plt.xlim(-L, L)
plt.grid(True)
plt.legend()
plt.show()

print("E (primeros 6 valores):", E)

# Comparación con solución exacta: E_n = omega (n + 1/2)
E_exact = omega * (np.arange(k) + 0.5)
print("E exactos:", E_exact)
print("Error abs:", np.abs(E - E_exact))
# Residuo L2: ||H psi - E psi||_2
for i in range(k):
    psi = U[:, i]
    r = H @ psi - E[i] * psi
    err = np.sqrt(np.sum(r**2) * h)
    print(f"Residuo L2 estado {i}: {err:.3e}")