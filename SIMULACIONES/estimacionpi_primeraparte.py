# Autora: Yolanda Valencia Díaz
# Primera tarea de la asignatura de Modelización y Simulación de Sistemas Complejos
# Primer apartado: estimación del valor de pi bajo integración de Monte Carlo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de simulación MC en 4 dimensiones

def estimacion_pi_4D(N_puntos, valor_inicial_semilla):
    N_interior = 0
    R = 1.0 # radio de la hiperesfera usando en ortante [0,1]^4

    
    rng_local=np.random.default_rng(int(valor_inicial_semilla)) # generador local para cada repetición, Si vuelvo a usar la semilla original, obtengo siempre el mismo resultado.
    for _ in range(N_puntos):
        x1,x2,x3,x4=rng_local.random(4)

        # Condición para que el punto se encuentre dentor de la hiperesfera

        radio = x1**2 + x2**2 + x3**2 + x4**2

        if radio <= R**2:
            N_interior += 1

    # Estimación de pi en base al volumen de las figuras y los puntos

    ratio = N_interior / N_puntos
    estimacion_pi = np.sqrt(32.0 * ratio)

    return estimacion_pi, ratio

N_puntos = 10000000 # número de puntos
M_repeticiones = 200 # número de repeticiones
semilla_base = 123456 # semilla con la que comienza 
rng=np.random.default_rng(semilla_base) # generador de números aleatorios para las subsemillas
semillas=rng.integers(1,10**9,size=M_repeticiones)# genera subsemillas diferentes para cada repetición
pi_estimacion = []

print(f"--- Estimación de pi en 4D (N={N_puntos} puntos por repetición) ---")

for i in range(M_repeticiones):
    valor_inicial_semilla = int(semillas[i])
    pi_est, ratio_est = estimacion_pi_4D(N_puntos, valor_inicial_semilla)
    pi_estimacion.append(pi_est)

    print(f"Repetición {i+1}/{M_repeticiones}: pi estimado = {pi_est:.9f}")

pi_medio = np.mean(pi_estimacion)

varianza = np.var(pi_estimacion, ddof = 1)
desviacion_estandar = np.std(pi_estimacion, ddof = 1)

# RESULTADOS DE LA PRUEBA

print("\n" + "="*70)
print(f"ESTIMACIÓN FINAL DE π (Promedio de M={M_repeticiones} simulaciones)")
print(f"Media de la Estimación de π: {pi_medio:.9f}")
print(f"Varianza: {varianza:.9f}")
print(f"Desviación Estándar Muestral: {desviacion_estandar:.9f}")
print(f"Valor Teórico de π: {np.pi:.9f}")
print("="*70)
