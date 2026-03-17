import numpy as np
import matplotlib.pyplot as plt
import time
#tiempo de ejecución INICIO
start_time = time.time()
# Parámetros
omega0 = 2.0
gamma  = 0.2
x0 = 1.0
v0 = 0.0
# Tiempo
t0, tf = 0.0, 20.0
N = 4000
t = np.linspace(t0, tf, N+1)
h = t[1] - t[0]
# Sistema 1er orden
def f(t, y):
    x, v = y
    dxdt = v
    dvdt = -2*gamma*v - omega0**2 * x
    return np.array([dxdt, dvdt])
# runge-kutta 4
y = np.zeros((N+1, 2))
y[0] = [x0, v0]
for i in range(N):
    k1 = f(t[i],         y[i])
    k2 = f(t[i] + h/2.0, y[i] + h*k1/2.0)
    k3 = f(t[i] + h/2.0, y[i] + h*k2/2.0)
    k4 = f(t[i] + h,     y[i] + h*k3)
    y[i+1] = y[i] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

x = y[:, 0]

# Solución analítica (subamortiguado)
wd = np.sqrt(max(omega0**2 - gamma**2, 0.0))
x_analitica = np.exp(-gamma*t) * (x0*np.cos(wd*t) + (v0 + gamma*x0)/wd*np.sin(wd*t))
# Error L2
err_L2 = np.sqrt(np.sum((x - x_analitica)**2) * h)
rel_L2 = err_L2 / (np.sqrt(np.sum(x_analitica**2)*h) + 1e-30)
print("Error L2 absoluto:", err_L2)
print("Error L2 relativo:", rel_L2)
#tiempo de ejecución FINAL
end_time = time.time()
print("Tiempo de ejecución:", end_time - start_time, "segundos")
# Gráfica de resultados
plt.figure(figsize=(10,5))
plt.plot(t, x, label="RK4")
plt.plot(t, x_analitica, "--", label="Analítica")
plt.title("Oscilador armónico amortiguado (RK4)")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)
plt.legend()
plt.show()