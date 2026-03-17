# Autora: Yolanda Valencia Díaz
# Primera tarea de la asignatura de Modelización y Simulación de Sistemas Complejos
# Estimación de π en 4D por Monte Carlo con variables antitéticas

import numpy as np
from concurrent.futures import ProcessPoolExecutor

# --------- Configuración ----------
N_puntos        = 10_000_000      # debe ser par (antitéticos)
M_repeticiones  =20000
semilla_base    = 123456
usar_paralelo   = True            # pon False si no quieres multiprocessing
chunk           = 2_000_000       # tamaño de bloque (ajusta a tu RAM)
# ----------------------------------

def estimacion_pi_4D_antitetico(N_puntos, valor_inicial_semilla, chunk=2_000_000):
    """
    Estimación de π en 4D con antitéticos:
      - Por cada X ~ U([0,1]^4), también evalúa Y = 1 - X.
      - Vectorizado por bloques (chunk) para velocidad y control de memoria.
      - Requiere N_puntos par.
    """
    if N_puntos % 2 != 0:
        raise ValueError("N_puntos debe ser par para usar antitéticos.")

    rng = np.random.default_rng(int(valor_inicial_semilla))

    dentro = 0
    # Trabajamos con N/2 parejas (X, 1-X)
    restantes = N_puntos // 2
    while restantes > 0:
        # cada iteración procesa 'm' parejas → 2*m puntos totales
        m = min(restantes, max(1, chunk // 2))

        # Genera X; Y = 1 - X (no crea distribución extra)
        # Usa float32 si tu NumPy lo soporta para ganar ancho de banda
        try:
            X = rng.random((m, 4), dtype=np.float32)
            Y = 1.0 - X

            # eleva al cuadrado in-place y suma por filas
            np.square(X, out=X)
            np.square(Y, out=Y)
            sX = X.sum(axis=1, dtype=np.float32)
            sY = Y.sum(axis=1, dtype=np.float32)
        except TypeError:
            # fallback a float64 si tu versión de NumPy no soporta dtype en random()
            X = rng.random((m, 4))
            Y = 1.0 - X
            np.square(X, out=X)
            np.square(Y, out=Y)
            sX = X.sum(axis=1)
            sY = Y.sum(axis=1)

        dentro += np.count_nonzero(sX <= 1.0)
        dentro += np.count_nonzero(sY <= 1.0)
        restantes -= m

    p = dentro / N_puntos
    pi_hat = np.sqrt(32.0 * p)  # p = π^2/32  ⇒ π = sqrt(32*p)
    return pi_hat, p

def _run_rep(seed):
    pi_est, _ = estimacion_pi_4D_antitetico(N_puntos, seed, chunk=chunk)
    return pi_est

def main():
    # Semillas independientes para cada repetición (a partir de una semilla maestra)
    rng_master = np.random.default_rng(semilla_base)
    semillas = rng_master.integers(1, 10**9, size=M_repeticiones)

    if usar_paralelo:
        with ProcessPoolExecutor() as ex:
            pi_estimacion = list(ex.map(_run_rep, semillas, chunksize=1))
    else:
        pi_estimacion = []
        for seed in semillas:
            pi_est, _ = estimacion_pi_4D_antitetico(N_puntos, int(seed), chunk=chunk)
            pi_estimacion.append(pi_est)

    pi_estimacion = np.array(pi_estimacion, dtype=float)
    pi_medio = pi_estimacion.mean()
    varianza = pi_estimacion.var(ddof=1)
    desviacion = pi_estimacion.std(ddof=1)

    print(f"--- Estimación de π en 4D con antitéticos ---")
    print(f"N por repetición: {N_puntos:,}")
    print(f"Repeticiones:     {M_repeticiones}")
    print("\n" + "="*70)
    print(f"ESTIMACIÓN FINAL DE π (promedio)")
    print(f"Media:                {pi_medio:.9f}")
    print(f"Varianza muestral:    {varianza:.9e}")
    print(f"Desv. estándar mues.: {desviacion:.9f}")
    print(f"Valor teórico π:      {np.pi:.9f}")
    print("="*70)

if __name__ == "__main__":
    main()
