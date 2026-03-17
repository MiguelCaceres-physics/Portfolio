import numpy as np

# =======================
# Parámetros del sistema
# =======================
Z = 2.0   # Helio
alpha = 1.7
beta  = 0.3
B     = 0.5

# Paso del Metropolis y paso del Laplaciano numérico
step_size = 0.5     # tamaño típico de movimiento (a.u.)
h_lap     = 1e-3    # paso para derivada segunda numérica

# =======================
# Función de onda trial
# =======================
def psi(R1, R2, alpha, beta, B):
    """
    R1, R2: arrays de longitud 3 (posición de cada electrón en 3D)
    """
    r1  = np.linalg.norm(R1)
    r2  = np.linalg.norm(R2)
    r12 = np.linalg.norm(R1 - R2)

    return np.exp(-alpha * (r1 + r2)) * (1.0 + B * np.exp(-beta * r12))


# =====================================
# Laplaciano numérico de la función de onda
# =====================================
def laplacian_psi(R1, R2, alpha, beta, B, h):
    """
    Calcula ∇1^2 psi + ∇2^2 psi mediante diferencias finitas centrales.
    """
    psi0 = psi(R1, R2, alpha, beta, B)
    lap = 0.0

    # Contribución del electrón 1
    for dim in range(3):
        e = np.zeros(3)
        e[dim] = 1.0

        R1_plus  = R1 + h * e
        R1_minus = R1 - h * e

        psi_plus  = psi(R1_plus,  R2, alpha, beta, B)
        psi_minus = psi(R1_minus, R2, alpha, beta, B)

        lap += (psi_plus + psi_minus - 2.0 * psi0) / h**2

    # Contribución del electrón 2
    for dim in range(3):
        e = np.zeros(3)
        e[dim] = 1.0

        R2_plus  = R2 + h * e
        R2_minus = R2 - h * e

        psi_plus  = psi(R1, R2_plus,  alpha, beta, B)
        psi_minus = psi(R1, R2_minus, alpha, beta, B)

        lap += (psi_plus + psi_minus - 2.0 * psi0) / h**2

    return lap


# ===========================
# Energía local E_loc = (Hψ)/ψ
# ===========================
def local_energy(R1, R2, alpha, beta, B, Z, h):
    psi0 = psi(R1, R2, alpha, beta, B)

    # cinética: -1/2 (∇1^2 + ∇2^2) ψ / ψ
    lap = laplacian_psi(R1, R2, alpha, beta, B, h)
    kinetic = -0.5 * lap / psi0

    # potencial: -Z/r1 - Z/r2 + 1/r12
    r1  = np.linalg.norm(R1)
    r2  = np.linalg.norm(R2)
    r12 = np.linalg.norm(R1 - R2)

    potential = -Z / r1 - Z / r2 + 1.0 / r12

    return kinetic + potential


# ===========================
# Algoritmo Metropolis VMC
# ===========================
def vmc_energy(alpha, beta, B, Z=2.0,
               n_steps=200_000, equilibration=20_000,
               step_size=0.5, h_lap=1e-3):
    """
    Devuelve <E> y el error estadístico (desviación estándar de la media).
    """

    # Posiciones iniciales (cerca del núcleo)
    R1 = np.random.normal(scale=1.0, size=3)
    R2 = np.random.normal(scale=1.0, size=3)

    psi_old = psi(R1, R2, alpha, beta, B)
    p_old   = psi_old**2

    energies = []
    acc = 0
    total_moves = 0

    for step in range(n_steps + equilibration):

        # Propuesta de movimiento (ambos electrones a la vez)
        dR1 = (np.random.rand(3) - 0.5) * step_size
        dR2 = (np.random.rand(3) - 0.5) * step_size

        R1_new = R1 + dR1
        R2_new = R2 + dR2

        psi_new = psi(R1_new, R2_new, alpha, beta, B)
        p_new   = psi_new**2

        # Criterio de aceptación Metropolis
        A = p_new / p_old
        if A >= 1.0 or np.random.rand() < A:
            # aceptar
            R1, R2 = R1_new, R2_new
            psi_old, p_old = psi_new, p_new
            acc += 1
        total_moves += 1

        # después de la equilibración empezamos a medir
        if step >= equilibration:
            E_loc = local_energy(R1, R2, alpha, beta, B, Z, h_lap)
            energies.append(E_loc)

    energies = np.array(energies)
    E_mean = np.mean(energies)
    E_err  = np.std(energies) / np.sqrt(len(energies))

    acc_rate = acc / total_moves
    print(f"Tasa de aceptación ≈ {acc_rate:.3f}")

    return E_mean, E_err


# ===========================
# Ejemplo de uso
# ===========================
if __name__ == "__main__":
    E, dE = vmc_energy(alpha, beta, B, Z=Z,
                       n_steps=100_000,
                       equilibration=10_000,
                       step_size=0.5,
                       h_lap=h_lap)
    print(f"Energia variacional ≈ {E:.6f} ± {dE:.6f} a.u.")
