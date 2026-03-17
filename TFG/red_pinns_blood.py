import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Carga de datos he copiado la carga del paquete del script de github, el formato de los datos es el mismo que el del script original.
data_mat = loadmat("TFG/DATA_HFM/Cylinder2D.mat")

U_star = data_mat["U_star"]  # (30189, 201)
V_star = data_mat["V_star"]  # (30189, 201)
P_star = data_mat["P_star"]  # (30189, 201)
t_star = data_mat["t_star"]  # (201, 1)
X_star = data_mat["x_star"]  # (30189, 1)
Y_star = data_mat["y_star"]  # (30189, 1)

N = X_star.shape[0]   # 30189
Nt = t_star.shape[0]  # 201 

XX = np.tile(X_star,    (1, Nt))   # (30189, 201)
YY = np.tile(Y_star,    (1, Nt))   # (30189, 201)
TT = np.tile(t_star.T,  (N, 1))    # (30189, 201)

x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]
u = U_star.flatten()[:, None]
v = V_star.flatten()[:, None]
p = P_star.flatten()[:, None]

data_all = np.concatenate([x, y, t, u, v, p], axis=1)
mask = (
    (data_all[:, 2] <= 7)  &
    (data_all[:, 0] >= 1)  &
    (data_all[:, 0] <= 8)  &
    (data_all[:, 1] >= -2) &
    (data_all[:, 1] <= 2)
)
data_domain = data_all[mask]
print(f"Puntos en dominio: {data_domain.shape[0]}")
np.random.seed(42)
num = 7000
idx = np.random.choice(data_domain.shape[0], num, replace=False)
x_train = data_domain[idx, 0:1]
y_train = data_domain[idx, 1:2]
t_train = data_domain[idx, 2:3]
u_train = data_domain[idx, 3:4]
v_train = data_domain[idx, 4:5]
p_train = data_domain[idx, 5:6]
X_train = np.hstack([x_train,y_train,t_train]) 
# Parámetros físicos
R     = 2.0   # radio (cm)
L     = 8.0   # longitud (cm)
T_end = 7.0   # periodo cardíaco (s)


C1 = dde.Variable(1.0)    # densidad en g/cm³ = 1000 kg/m³
C2 = dde.Variable(0.004)  # viscosidad en g/(cm·s) = Pa·s

# Dominio, voy a suponer el dominio rectangular para simular el flujo en la arteria/vena
geom= dde.geometry.Rectangle(xmin=[1, -2], xmax=[L, R])
time_domain= dde.geometry.TimeDomain(0, T_end)
geomtime= dde.geometry.GeometryXTime(geom, time_domain)

# Navier-Stokes
def Navier_Stokes_Equation(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    u_t  = dde.grad.jacobian(y, x, i=0, j=2)
    u_x  = dde.grad.jacobian(y, x, i=0, j=0)
    u_y  = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_t  = dde.grad.jacobian(y, x, i=1, j=2)
    v_x  = dde.grad.jacobian(y, x, i=1, j=0)
    v_y  = dde.grad.jacobian(y, x, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    p_x  = dde.grad.jacobian(y, x, i=2, j=0)
    p_y  = dde.grad.jacobian(y, x, i=2, j=1)
    continuidad = u_x + v_y
    x_momento   = C1*(u_t + u*u_x + v*u_y) + p_x - C2*(u_xx + u_yy)
    y_momento   = C1*(v_t+u*v_x + v*v_y) + p_y - C2*(v_xx + v_yy)
    return [continuidad, x_momento, y_momento]

# Condiciones de frontera y condiciones iniciales, suponemos del tipo Dirichlet para la velocidad y presión en la entrada/salida y paredes, y condiciones iniciales de reposo
zeros   = lambda x: np.zeros((x.shape[0], 1))
boundary_inlet   = lambda x, on: on and np.isclose(x[0], 1.0)
boundary_outlet  = lambda x, on: on and np.isclose(x[0], L)
boundary_walls   = lambda x, on: on and (np.isclose(x[1], R) or np.isclose(x[1], -R))
bc_wall_u   = dde.DirichletBC(geomtime, zeros,   boundary_walls,  component=0)
bc_wall_v   = dde.DirichletBC(geomtime, zeros,   boundary_walls,  component=1)
bc_inlet_v  = dde.DirichletBC(geomtime, zeros,   boundary_inlet,  component=1)
bc_outlet_p = dde.DirichletBC(geomtime, zeros,   boundary_outlet, component=2)

# Condición inicial: fluido en reposo
ic_u = dde.IC(geomtime, zeros, lambda x, on: on, component=0)
ic_v = dde.IC(geomtime, zeros, lambda x, on: on, component=1)

# Observaciones de los datos reales
observe_u = dde.PointSetBC(X_train, u_train, component=0)
observe_v = dde.PointSetBC(X_train, v_train, component=1)

# Problema y datos para entrenamiento del PINN, en numero de puntos de dominio, frontera e iniciales son completamente aleatorios, voy ajustando para la convergencia
data = dde.data.TimePDE(geomtime,Navier_Stokes_Equation,[bc_wall_u, bc_wall_v,bc_inlet_v,bc_outlet_p,ic_u, ic_v,observe_u, observe_v],num_domain=10000,num_boundary=2000,num_initial=2000,anchors=X_train)

# Red y entrenamiento, la arquitectura de la red reconoce 3 capas ocultas con 65 neuronas cada una, función de activación tangente hiperbólica y pesos de inicialización Glorot normal.
net = dde.nn.FNN([3] + [64] * 3 + [3], "tanh", "Glorot normal")
model = dde.Model(data, net)
# se pueden utilizar otros optimizadores estoy probando L-BFGS para refinamiento
model.compile("adam",lr=1e-3,external_trainable_variables=[C1, C2])  #identificación de parámetros

losshistory, train_state = model.train(epochs=20000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Resultados
print(f"Densidad   C1 = {dde.backend.to_numpy(C1):.4f} g/cm³")
print(f"Viscosidad C2 = {dde.backend.to_numpy(C2):.6f} Pa·s")
#validación sobre puntos observados y errores L2 y MAE
pred_train = model.predict(X_train)
u_pred_train = pred_train[:,0:1]
v_pred_train = pred_train[:,1:2]
p_pred_train = pred_train[:,2:3]
#errores
error_u = np.linalg.norm(u_pred_train-u_train,2)/np.linalg.norm(u_train,2)
error_v = np.linalg.norm(v_pred_train-v_train,2)/np.linalg.norm(v_train,2)
error_p = np.linalg.norm(p_pred_train-p_train,2)/np.linalg.norm(p_train,2)
MAE_p = np.mean(np.abs(p_pred_train-p_train))
print(f"Error relativo L2 en u: {error_u:.6e}")
print(f"Error relativo L2 en v: {error_v:.6e}")
print(f"Error relativo L2 en p: {error_p:.6e}")
print(f"MAE de la presión: {MAE_p:.6e}")
# visualización
x_test  = np.linspace(1, L, 100)
y_test  = np.linspace(-R, R, 50)
X_grid, Y_grid = np.meshgrid(x_test, y_test)
t_fixed = 0.5
test_points = np.hstack([X_grid.flatten()[:, None],Y_grid.flatten()[:, None],np.full((X_grid.size, 1), t_fixed)])
pred   = model.predict(test_points)
u_pred = np.array(pred[:, 0]).reshape(X_grid.shape)
v_pred = np.array(pred[:, 1]).reshape(X_grid.shape)
p_pred = np.array(pred[:, 2]).reshape(X_grid.shape)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
campos  = [u_pred, v_pred, p_pred]
titulos = ["u (velocidad x)", "v (velocidad y)", "p (presión)"]
for ax, campo, titulo in zip(axes, campos, titulos):
    c = ax.contourf(X_grid, Y_grid, campo, levels=50, cmap="RdBu_r")
    plt.colorbar(c, ax=ax)
    ax.set_title(f"{titulo}   t = {t_fixed} s")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
plt.suptitle("Campos predichos por el PINN", y=1.02)
plt.tight_layout()
plt.show()