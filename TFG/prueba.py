import os
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

print("Backend:", dde.backend.backend_name)
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ═══════════════════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════
data_mat = loadmat("TFG/DATA_HFM/Cylinder2D.mat")

U_star = data_mat["U_star"]  # (30189, 201)
V_star = data_mat["V_star"]  # (30189, 201)
P_star = data_mat["P_star"]  # (30189, 201)
t_star = data_mat["t_star"]  # (201, 1)
X_star = data_mat["x_star"]  # (30189, 1)
Y_star = data_mat["y_star"]  # (30189, 1)

N  = X_star.shape[0]
Nt = t_star.shape[0]

XX = np.tile(X_star,   (1, Nt))
YY = np.tile(Y_star,   (1, Nt))
TT = np.tile(t_star.T, (N, 1))

x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]
u = U_star.flatten()[:, None]
v = V_star.flatten()[:, None]
p = P_star.flatten()[:, None]

# Filtrar dominio: X=[1,8], Y=[-2,2], T=[0,7]
data_all = np.concatenate([x, y, t, u, v, p], axis=1)
mask = (
    (data_all[:, 2] <= 7)  &
    (data_all[:, 0] >= 1)  &
    (data_all[:, 0] <= 8)  &
    (data_all[:, 1] >= -2) &
    (data_all[:, 1] <= 2)
)
data_domain = data_all[mask]
print(f"Puntos totales en dominio: {data_domain.shape[0]}")

# Selección aleatoria reproducible
np.random.seed(42)
num = 7000
idx = np.random.choice(data_domain.shape[0], num, replace=False)

x_train = data_domain[idx, 0:1]
y_train = data_domain[idx, 1:2]
t_train = data_domain[idx, 2:3]
u_train = data_domain[idx, 3:4]
v_train = data_domain[idx, 4:5]
p_train = data_domain[idx, 5:6]  # reservado para validación, NO entra en entrenamiento
X_train = np.hstack([x_train, y_train, t_train])

print(f"Puntos de entrenamiento: {num}")
print(f"Rango u: [{u_train.min():.3f}, {u_train.max():.3f}]")
print(f"Rango v: [{v_train.min():.3f}, {v_train.max():.3f}]")
print(f"Rango p: [{p_train.min():.3f}, {p_train.max():.3f}]")

# ═══════════════════════════════════════════════════════════════════
# 2. PARÁMETROS A IDENTIFICAR
# Formulación adimensional de Raissi et al. (2019):
#   u_t + λ1(u·u_x + v·u_y) + p_x - λ2(u_xx + u_yy) = 0
#   v_t + λ1(u·v_x + v·v_y) + p_y - λ2(v_xx + v_yy) = 0
#   u_x + v_y = 0
# Valores esperados: λ1 ≈ 1.0, λ2 ≈ 0.01 (Re ≈ 100)
# ═══════════════════════════════════════════════════════════════════
lambda1 = dde.Variable(1.0)   # término convectivo
lambda2 = dde.Variable(0.01)  # término difusivo = 1/Re

# ═══════════════════════════════════════════════════════════════════
# 3. DOMINIO ESPACIO-TEMPORAL
# ═══════════════════════════════════════════════════════════════════
geom        = dde.geometry.Rectangle(xmin=[1, -2], xmax=[8, 2])
time_domain = dde.geometry.TimeDomain(0, 7)
geomtime    = dde.geometry.GeometryXTime(geom, time_domain)

# ═══════════════════════════════════════════════════════════════════
# 4. ECUACIONES DE NAVIER-STOKES (formulación Raissi)
# ═══════════════════════════════════════════════════════════════════
def Navier_Stokes(x, y):
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
    x_momento   = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy)
    y_momento   = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)

    return [continuidad, x_momento, y_momento]

# ═══════════════════════════════════════════════════════════════════
# 5. CONDICIONES DE CONTORNO
# Solo las mínimas necesarias para anclar la solución
# ═══════════════════════════════════════════════════════════════════
zeros = lambda x: np.zeros((x.shape[0], 1))

# Paredes superior e inferior: no-slip (u=0, v=0)
boundary_walls  = lambda x, on: on and (np.isclose(x[1],  2.0) or
                                         np.isclose(x[1], -2.0))
# Inlet (x=1): solo v=0, u libre (lo aprende de los datos)
boundary_inlet  = lambda x, on: on and np.isclose(x[0], 1.0)
# Outlet (x=8): presión de referencia p=0
boundary_outlet = lambda x, on: on and np.isclose(x[0], 8.0)

bc_wall_u   = dde.DirichletBC(geomtime, zeros, boundary_walls,  component=0)
bc_wall_v   = dde.DirichletBC(geomtime, zeros, boundary_walls,  component=1)
bc_inlet_v  = dde.DirichletBC(geomtime, zeros, boundary_inlet,  component=1)
bc_outlet_p = dde.DirichletBC(geomtime, zeros, boundary_outlet, component=2)

# Condición inicial: fluido en reposo
ic_u = dde.IC(geomtime, zeros, lambda x, on: on, component=0)
ic_v = dde.IC(geomtime, zeros, lambda x, on: on, component=1)

# ═══════════════════════════════════════════════════════════════════
# 6. OBSERVACIONES — solo u y v, nunca p (problema inverso)
# ═══════════════════════════════════════════════════════════════════
observe_u = dde.PointSetBC(X_train, u_train, component=0)
observe_v = dde.PointSetBC(X_train, v_train, component=1)
# p_train NO se usa aquí → se infiere a través de las NS

# ═══════════════════════════════════════════════════════════════════
# 7. PROBLEMA COMPLETO
# ═══════════════════════════════════════════════════════════════════
data = dde.data.TimePDE(
    geomtime,
    Navier_Stokes,
    [bc_wall_u, bc_wall_v,
     bc_inlet_v,
     bc_outlet_p,
     ic_u, ic_v,
     observe_u, observe_v],
    num_domain=5000,
    num_boundary=1000,
    num_initial=1000,
    anchors=X_train
)

# ═══════════════════════════════════════════════════════════════════
# 8. RED NEURONAL Y ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════
# Arquitectura: (x,y,t) → (u,v,p)
# 6 capas × 64 neuronas, suficiente para capturar la estela de vórtices
net = dde.nn.FNN([3] + [64] * 6 + [3], "tanh", "Glorot normal")

model = dde.Model(data, net)

# Fase 1: Adam — exploración rápida del espacio de parámetros
model.compile(
    "adam",
    lr=1e-3,
    external_trainable_variables=[lambda1, lambda2],
    loss_weights=[1, 1, 1,      # continuidad, x_momento, y_momento
                  1, 1, 1, 1,   # bc_wall_u, bc_wall_v, bc_inlet_v, bc_outlet_p
                  1, 1,          # ic_u, ic_v
                  10, 10]        # observe_u, observe_v — más peso a los datos
)
print("\n── Fase 1: Adam (20000 epochs) ──────────────────────────")
losshistory, train_state = model.train(epochs=20000, display_every=1000)

# Fase 2: L-BFGS — refinamiento de precisión
print("\n── Fase 2: L-BFGS ───────────────────────────────────────")
model.compile("L-BFGS", external_trainable_variables=[lambda1, lambda2])
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# ═══════════════════════════════════════════════════════════════════
# 9. PARÁMETROS IDENTIFICADOS
# ═══════════════════════════════════════════════════════════════════
l1 = float(dde.backend.to_numpy(lambda1))
l2 = float(dde.backend.to_numpy(lambda2))
Re = 1.0 / l2 if l2 != 0 else float("inf")

print("\n╔══════════════════════════════════════════╗")
print("║      PARÁMETROS IDENTIFICADOS            ║")
print("╠══════════════════════════════════════════╣")
print(f"║  λ1 = {l1:.6f}   (esperado ≈ 1.000000)  ║")
print(f"║  λ2 = {l2:.6f}   (esperado ≈ 0.010000)  ║")
print(f"║  Re = {Re:.2f}       (esperado ≈ 100.00)    ║")
print("╚══════════════════════════════════════════╝")

# ═══════════════════════════════════════════════════════════════════
# 10. VALIDACIÓN — p inferida vs p real (que nunca se usó)
# ═══════════════════════════════════════════════════════════════════
pred_train   = model.predict(X_train)
u_pred_train = pred_train[:, 0:1]
v_pred_train = pred_train[:, 1:2]
p_pred_train = pred_train[:, 2:3]

error_u = np.linalg.norm(u_pred_train - u_train) / np.linalg.norm(u_train)
error_v = np.linalg.norm(v_pred_train - v_train) / np.linalg.norm(v_train)
error_p = np.linalg.norm(p_pred_train - p_train) / np.linalg.norm(p_train)
MAE_p   = np.mean(np.abs(p_pred_train - p_train))

print("\n── Errores de validación ────────────────────────────────")
print(f"  L2 u : {error_u:.4e}  (observable,    ref. < 0.05)")
print(f"  L2 v : {error_v:.4e}  (observable,    ref. < 0.05)")
print(f"  L2 p : {error_p:.4e}  (INFERIDA,      ref. < 0.15)")
print(f"  MAE p: {MAE_p:.4e}")

# ═══════════════════════════════════════════════════════════════════
# 11. FIGURA 1 — Campos predichos en malla (t=4s)
# ═══════════════════════════════════════════════════════════════════
x_test = np.linspace(1, 8, 150)
y_test = np.linspace(-2, 2, 75)
X_grid, Y_grid = np.meshgrid(x_test, y_test)
t_fixed = 4.0

test_points = np.hstack([
    X_grid.flatten()[:, None],
    Y_grid.flatten()[:, None],
    np.full((X_grid.size, 1), t_fixed)
])

pred   = model.predict(test_points)
u_pred = np.array(pred[:, 0]).reshape(X_grid.shape)
v_pred = np.array(pred[:, 1]).reshape(X_grid.shape)
p_pred = np.array(pred[:, 2]).reshape(X_grid.shape)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, campo, titulo, cmap in zip(
    axes,
    [u_pred,            v_pred,            p_pred],
    ["u  velocidad x",  "v  velocidad y",  "p  presión (INFERIDA)"],
    ["RdBu_r",          "RdBu_r",          "RdBu_r"]
):
    c = ax.contourf(X_grid, Y_grid, campo, levels=60, cmap=cmap)
    plt.colorbar(c, ax=ax)
    ax.set_title(f"{titulo}   t = {t_fixed} s", fontsize=11)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

plt.suptitle(
    f"PINN — campos predichos   |   λ1={l1:.3f}  λ2={l2:.4f}  Re={Re:.1f}",
    fontsize=12, y=1.02
)
plt.tight_layout()
plt.savefig("fig1_campos_predichos.png", dpi=150, bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 12. FIGURA 2 — Comparación p inferida vs p real en puntos de datos
# ═══════════════════════════════════════════════════════════════════
mask_t  = np.isclose(t_train.flatten(), t_fixed, atol=0.2)
x_sel   = x_train[mask_t]
y_sel   = y_train[mask_t]
p_real  = p_train[mask_t]
p_infer = p_pred_train[mask_t]

vmin = min(p_real.min(), p_infer.min())
vmax = max(p_real.max(), p_infer.max())

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sc1 = axes[0].scatter(x_sel, y_sel, c=p_real,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax, s=20)
axes[0].set_title("p  REAL\n(nunca usada en entrenamiento)", fontsize=11)
plt.colorbar(sc1, ax=axes[0])

sc2 = axes[1].scatter(x_sel, y_sel, c=p_infer,
                       cmap="RdBu_r", vmin=vmin, vmax=vmax, s=20)
axes[1].set_title("p  INFERIDA por PINN\n(solo con u, v y ecuaciones NS)", fontsize=11)
plt.colorbar(sc2, ax=axes[1])

error_abs = np.abs(p_real - p_infer)
sc3 = axes[2].scatter(x_sel, y_sel, c=error_abs,
                       cmap="hot_r", s=20)
axes[2].set_title(f"Error absoluto |p_real − p_infer|\nMAE = {MAE_p:.4e}", fontsize=11)
plt.colorbar(sc3, ax=axes[2])

for ax in axes:
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

plt.suptitle(
    f"Validación inferencia de presión   t ≈ {t_fixed} s   |   L2 = {error_p:.4e}",
    fontsize=12
)
plt.tight_layout()
plt.savefig("fig2_comparacion_presion.png", dpi=150, bbox_inches="tight")
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 13. FIGURA 3 — Evolución temporal de la presión inferida
# ═══════════════════════════════════════════════════════════════════
t_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for ax, t_val in zip(axes, t_vals):
    pts = np.hstack([
        X_grid.flatten()[:, None],
        Y_grid.flatten()[:, None],
        np.full((X_grid.size, 1), t_val)
    ])
    pr = np.array(model.predict(pts)[:, 2]).reshape(X_grid.shape)
    c  = ax.contourf(X_grid, Y_grid, pr, levels=50, cmap="RdBu_r")
    plt.colorbar(c, ax=ax)
    ax.set_title(f"p inferida   t = {t_val} s")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")

plt.suptitle("Evolución temporal de la presión inferida por PINN", fontsize=13)
plt.tight_layout()
plt.savefig("fig3_evolucion_temporal_p.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Guardar modelo y resultados ───────────────────────────────────
os.makedirs("resultados_pinn", exist_ok=True)
model.save("resultados_pinn/modelo_final")
np.savez("resultados_pinn/datos_entrenamiento.npz",
         x_train=x_train, y_train=y_train, t_train=t_train,
         u_train=u_train, v_train=v_train, p_train=p_train)
np.save("resultados_pinn/parametros.npy", np.array([l1, l2]))
print("\n✅ Modelo y resultados guardados en 'resultados_pinn/'")