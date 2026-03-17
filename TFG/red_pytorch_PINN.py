import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
#parametros
omega0 = 1.0
gamma = 0.1
x0 = 1.0
v0 = 0.0
#ecuación ocilador armonico submamortiguado#
def solucion_analitica(t, omega0, gamma, x0, v0):
    omega_d = torch.sqrt(torch.tensor(omega0**2 - gamma**2, dtype=t.dtype))
    return torch.exp(-gamma*t) * ( x0*torch.cos(omega_d*t) + (v0 + gamma*x0)/omega_d * torch.sin(omega_d*t) )

#RED NEURONAL#
class PINN(nn.Module):
    def __init__(self,N_input,N_oculta,N_output,N_capas):
        super().__init__()
        self.funcion_activacion = nn.Tanh()
        self.capa_input = nn.Sequential(nn.Linear(N_input,N_oculta), self.funcion_activacion)
        self.capa_oculta = nn.Sequential(*[nn.Sequential(nn.Linear(N_oculta,N_oculta), self.funcion_activacion) for _ in range(N_capas-1)])
        self.capa_output = nn.Linear(N_oculta,N_output)
    
    def forward(self,x):
        x = self.capa_input(x)
        x = self.capa_oculta(x)
        return self.capa_output(x)

#Puntos para el modelo PINN temporales#
t = torch.linspace(0, 20, 4000).reshape(-1, 1).requires_grad_(True)
#fijamos la seed para la reproducibilidad
torch.manual_seed(42)
#Inicializamos la red neuronal
model = PINN(N_input=1, N_oculta=50, N_output=1, N_capas=3)
#Definimos el optimizador# #Recordar probar en Adam y SGD para comparar resultados#
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_values = []
#Entrenamiento#
import time; start_time = time.time()
for epoch in range(30000):
    optimizer.zero_grad()
    #Predicción de la red neuronal#
    x_pred = model(t)
    # Primera derivada (velocidad)
    dx_dt = torch.autograd.grad(x_pred, t, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
    # Segunda derivada (aceleración)
    d2x_dt2 = torch.autograd.grad(dx_dt, t, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]  
    #Definimos la función de pérdida#
    f = d2x_dt2 + 2*gamma*dx_dt + omega0**2*x_pred
    loss_f = torch.mean(f**2)
    # Condición inicial x(0) = x0
    x_0_pred = model(torch.tensor([[0.0]]))
    loss_u = torch.mean((x_0_pred - x0)**2)
    # Condición de velocidad inicial v(0) = 0
    t0 = torch.tensor([[0.0]], requires_grad=True)
    x0_val = model(t0)
    v0_val = torch.autograd.grad(x0_val, t0, create_graph=True)[0]
    loss_v = torch.mean((v0_val - v0)**2)
    # Pérdida total con pesos para cada término (al final deje lo pesos en la unidad para no complicar el análisis, pero se pueden ajustar para mejorar la convergencia)
    loss = loss_u + loss_f + loss_v
    loss_values.append(loss.item())
    loss.backward()
    optimizer.step()    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
#evaluación del modelo#
X_analitica=solucion_analitica(t,omega0, gamma, x0, v0)
x_pred = model(t)
#calculo de norma L2
error_relativo_L2=torch.sqrt(torch.mean((x_pred - X_analitica)**2)) / torch.sqrt(torch.mean(X_analitica**2))
print(f"Error relativo L2: {error_relativo_L2.item():.6f}")
#error absoluto medio MAE
error_absoluto_medio_MAE=torch.mean(torch.abs(x_pred - X_analitica))
print(f"Error absoluto medio MAE: {error_absoluto_medio_MAE.item():.6f}")
#estabilidad del método
estabilidad_final = np.var(loss_values[-100:])
print(f"Estabilidad del método (varianza de la pérdida en las últimas 100 épocas): {estabilidad_final:.6e}")
print(f"Tiempo de cálculo: {time.time() - start_time:.2f} segundos")
#Resultados#
x_pred = model(t).detach().numpy()
t = t.detach().numpy()
plt.figure(figsize=(10, 6))
plt.plot(t, x_pred, label='PINN Predicción', color='red')
#Guardamos la configuración de pesos y sesgos
torch.save(model.state_dict(), "pinn_oscilador_subamortiguado_PINNs.pth")
#guardamos datos para comparación con RK4 y red neuronal básica
np.savez("pinn_oscilador_subamortiguado_PINNs.npz", t=t, x_pred=x_pred, error_absoluto_medio_MAE=error_absoluto_medio_MAE.item(), error_relativo_L2=error_relativo_L2.item(), estabilidad_final=estabilidad_final, loss_values=loss_values)
# Solución analítica para comparación
wd = np.sqrt(max(omega0**2 - gamma**2, 0.0))
t_analitica = np.linspace(0, 20, 1000)
x_analitica = np.exp(-gamma*t_analitica)*(x0*np.cos(wd*t_analitica)+(v0+gamma*x0)/wd*np.sin(wd*t_analitica))
plt.plot(t_analitica, x_analitica , label='Solución Analítica', color='blue', linestyle='dashed')
plt.scatter(0, x0, color='green', label='Condición Inicial', zorder=5) 
plt.title('Comparación entre Solución Analítica y PINN')
plt.xlabel('t(s)')
plt.ylabel('x(t)')
plt.legend()
plt.grid()
#grafica de la función de pérdida#
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Pérdida Total', color='purple')
plt.title('Evolución de la fución de pérdida durante el entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()
