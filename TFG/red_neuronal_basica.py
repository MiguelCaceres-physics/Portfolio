import numpy as np
import matplotlib.pyplot as plt 
# Parámetros
omega0 = 1.0
gamma  = 0.1
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
#Vamos a crear una red neuronal con tres capas ocultas de 50 neuronas, 1 capa de entrada y una capa de salida
np.random.seed(42)
#Pesos de la capa de entrada a la primera capa oculta
W1 = np.random.randn(1,50)*np.sqrt(1/1)
b1 = np.zeros((1,50))
#Pesos de la primera capa oculta a la segunda capa oculta
W2 = np.random.randn(50,50)*np.sqrt(1/50)
b2 = np.zeros((1,50))
#Pesos de la segunda capa oculta a la capa de salida
W3 = np.random.randn(50,50)*np.sqrt(1/50)
b3 = np.zeros((1,50))
#Pesos de la tercera capa oculta a la capa de salida
W4 = np.random.randn(50,1)*np.sqrt(1/50)
b4 = np.zeros((1,1))
#Función de activación tangente hiperbólica (evitamos la sigmoide al no reflejar valores negativos)
def tanh(x):
    return np.tanh(x)
def tanh_derivada(x):
    return 1.0 - np.tanh(x)**2
#Bucle de entrenamiento
learning_rate = 0.15
epochs = 100000
for epoch in range(epochs):
    #Forward pass (adelante)
    z1 = X.dot(W1) + b1
    a1 = tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = tanh(z2)
    z3 = a2.dot(W3) + b3
    a3 = tanh(z3)
    z4 = a3.dot(W4) + b4
    y_pred = z4
    #Cálculo del error (MSE)
    error = np.mean((y_pred - y)**2)
    #Backward pass (retropropagación)
    derror_dypred = 2*(y_pred - y)/y.shape[0]
    dypred_dz4 = 1
    dz4_da3 = W4
    da3_dz3 = tanh_derivada(z3)
    dz3_da2 = W3
    da2_dz2 = tanh_derivada(z2)
    dz2_da1 = W2
    da1_dz1 = tanh_derivada(z1)
    dz1_dW1 = X
    #Gradientes para la capa de salida
    derror_dz4 = derror_dypred * dypred_dz4
    derror_dW4 = a3.T.dot(derror_dz4)
    derror_db4 = np.sum(derror_dz4, axis=0, keepdims=True)
    #Gradientes para la tercera capa oculta
    derror_da3 = derror_dz4.dot(dz4_da3.T)
    derror_dz3 = derror_da3 * da3_dz3
    derror_dW3 = a2.T.dot(derror_dz3)
    derror_db3 = np.sum(derror_dz3, axis=0, keepdims=True)
    #Gradientes para la segunda capa oculta
    derror_da2 = derror_dz3.dot(dz3_da2.T)
    derror_dz2 = derror_da2 * da2_dz2
    derror_dW2 = a1.T.dot(derror_dz2)
    derror_db2 = np.sum(derror_dz2, axis=0, keepdims=True)
    #Gradientes para la primera capa oculta
    derror_da1 = derror_dz2.dot(dz2_da1.T)
    derror_dz1 = derror_da1 * da1_dz1
    derror_dW1 = X.T.dot(derror_dz1)
    derror_db1 = np.sum(derror_dz1, axis=0, keepdims=True)
    #Actualización de pesos y sesgos
    W4 -= learning_rate * derror_dW4
    b4 -= learning_rate * derror_db4
    W3 -= learning_rate * derror_dW3
    b3 -= learning_rate * derror_db3
    W2 -= learning_rate * derror_dW2
    b2 -= learning_rate * derror_db2
    W1 -= learning_rate * derror_dW1
    b1 -= learning_rate * derror_db1
    #Imprimir el error cada 1000 épocas
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")
#Predicción final
z1 = X.dot(W1) + b1
a1 = tanh(z1)
z2 = a1.dot(W2) + b2
a2 = tanh(z2)
z3 = a2.dot(W3) + b3  
a3 = tanh(z3)
z4 = a3.dot(W4) + b4
y_pred = z4
#guardar configuración de pesos y sesgos
np.savez("red_neuronal_osc_amortiguado.npz", W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, W4=W4, b4=b4)
#error relativo L2 entre la predicción y los datos reales
error_relativo = np.linalg.norm(y_pred - y) / np.linalg.norm(y)
print(f"Error relativo L2: {error_relativo:.6f}")
#error MAE entre la predicción y los datos reales
error_mae = np.mean(np.abs(y_pred - y))
print(f"Error MAE: {error_mae:.6f}")
#Gráfica de resultados
plt.figure(figsize=(10,5))
plt.plot(t, y, label="Datos Reales")
plt.plot(t, y_pred, label="Predicción Red Neuronal")   
plt.title("Red Neuronal para Oscilador Armónico Amortiguado")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.grid(True)
plt.legend()
plt.show()