import deepxde as dde
import numpy as np
from scipy.io import loadmat
data=loadmat("TFG/DATA_HFM/Cylinder2D.mat")
print(data.keys())
print(data["x_star"].shape)
print(data["y_star"].shape)
print(data["t_star"].shape)
print(data["U_star"].shape)
print(data["V_star"].shape)
print(data["P_star"].shape)