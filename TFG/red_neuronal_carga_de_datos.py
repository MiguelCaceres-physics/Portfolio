import numpy as np
import matplotlib.pyplot as plt 
# Parámetros
omega0 = 2.0
gamma  = 0.2
x0 = 1.0
v0 = 0.0
# Tiempo
t0, tf = 0.0, 20.0
N = 4000
t = np.linspace(t0, tf, N)
#solución analítica (subamortiguado)
wd = np.sqrt(max(omega0**2 - gamma**2, 0.0))
x_analitica = np.exp(-gamma*t) * (x0*np.cos(wd*t) + (v0 + gamma*x0)/wd*np.sin(wd*t))
#Vamos a normalizar los datos para que estén entre 0 y 1
X= t.reshape(-1,1)/np.max(t)
y = x_analitica.reshape(-1,1)
# Cargamos la red neuronal entrenada
data = np.load("TFG/DATA/SEGUNDA_RED/red_neuronal_tercer_refinamiento.npz")
W1 = data['W1']
b1 = data['b1']
W2 = data['W2']
b2 = data['b2']
W3 = data['W3']
b3 = data['b3']
#Función de activación tangente hiperbólica (evitamos la sigmoide al no reflejar valores negativos)
def tanh(x):
    return np.tanh(x)
def tanh_derivada(x):
    return 1.0 - np.tanh(x)**2
#Bucle de entrenamiento
learning_rate = 0.11
epochs = 30000
historial_error = []
for epoch in range(epochs):
    #Forward pass (adelante)
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = tanh(z2)
    z3 = a2.dot(W3) + b3
    y_pred = z3
    #Cálculo del error (MSE)
    error = np.mean((y_pred - y)**2)
    historial_error.append(error)
    #Backward pass (retropropagación)
    derror_dypred = 2*(y_pred - y)/y.shape[0]
    derror_dz3 = derror_dypred
    derror_da2 = derror_dz3.dot(W3.T)
    derror_dz2 = derror_da2 * tanh_derivada(z2)
    derror_da1 = derror_dz2.dot(W2.T)
    derror_dz1 = derror_da1 * tanh_derivada(z1)
    #Actualización de pesos y sesgos
    W3 -= learning_rate * (a2.T.dot(derror_dz3))
    b3 -= learning_rate * np.sum(derror_dz3, axis=0, keepdims=True)
    W2 -= learning_rate * (a1.T.dot(derror_dz2))
    b2 -= learning_rate * np.sum(derror_dz2, axis=0, keepdims=True)
    W1 -= learning_rate * (X.T.dot(derror_dz1))
    b1 -= learning_rate * np.sum(derror_dz1, axis=0, keepdims=True)
    #Imprimir el error cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")
#Predicción final
z1 = X.dot(W1) + b1
a1 = tanh(z1)
z2 = a1.dot(W2) + b2
a2 = tanh(z2)
z3 = a2.dot(W3) + b3  
y_pred = z3
#guardar configuración de pesos y sesgos
np.savez("red_neuronal_cuarto_refinamiento.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, mse_final=error, historial=np.array(historial_error))
#graficas Loss curve y función aproximada vs analítica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# Loss curve
ax1.plot(historial_error)
ax1.set_yscale('log')
ax1.set_title("Curva de error (MSE) durante el entrenamiento")
ax1.set_xlabel("Épocas")
ax1.set_ylabel("Error (MSE)")
ax1.grid()
# Función aproximada vs analítica
ax2.plot(t, y, label="Analítica")
ax2.plot(t, y_pred, label="Red Neuronal", linestyle='--')
ax2.set_title("Comparación entre función analítica y red neuronal")
ax2.set_xlabel("Tiempo (t)")
ax2.set_ylabel("x(t)")
ax2.legend()
ax2.grid()
plt.tight_layout()


plt.show()