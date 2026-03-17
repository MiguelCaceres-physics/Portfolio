import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
time_start = time.time()
# 1. Parámetros
omega0, gamma, x0, v0 = 2.0, 0.2, 1.0, 0.0
t0, tf, N = 0.0, 20.0, 4000

t_np = np.linspace(t0, tf, N)
wd = np.sqrt(max(omega0**2 - gamma**2, 0.0))
x_ana_np = np.exp(-gamma*t_np) * (x0*np.cos(wd*t_np) + (v0 + gamma*x0)/wd*np.sin(wd*t_np))

X = torch.tensor(t_np.reshape(-1, 1) / tf, dtype=torch.float32)
y = torch.tensor(x_ana_np.reshape(-1, 1), dtype=torch.float32)

# Arquitectura de la Red (segimos la misma que en la versión básica)
class RedOscilador(nn.Module):
    def __init__(self):
        super(RedOscilador, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# Seed y modelo
torch.manual_seed(42)
model = RedOscilador()
# Usamos el optimizador Adam y la función de pérdida MSE
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 50000 
loss_history = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(X)
    # Calcular pérdida
    loss = criterion(y_pred, y)
    # Backward pass
    loss.backward()
    # Actualizamos los pesos
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.8f}")
time_end = time.time()
print(f"\nTiempo de entrenamiento: {time_end - time_start:.2f} segundos")
# Evaluamos el modelo 
model.eval()
with torch.no_grad():
    y_final = model(X)
# Error relativo L2
error_l2 = torch.norm(y_final - y) / torch.norm(y)
# Error MAE
error_mae = torch.mean(torch.abs(y_final - y))

print(f"\nError relativo L2: {error_l2.item():.6f}")
print(f"Error MAE: {error_mae.item():.6f}")
#guardamos pesos, sesgos, perdidas, errores, tiempo de entrenamiento
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_history': loss_history,
    'error_l2': error_l2.item(),
    'error_mae': error_mae.item(),
    'training_time': time_end - time_start
}, 'red_oscilador.pth')
#graficamos solución y errores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_np, x_ana_np, label='Solución Analítica', color='blue')
plt.plot(t_np, y_final.numpy(), label='Red Neuronal', color='red', linestyle='dashed')
plt.title('Resolución del Oscilador Armónico Amortiguado con Red Neuronal')
plt.xlabel('Tiempo (s)')
plt.ylabel('x(t)')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Pérdida durante el entrenamiento')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
