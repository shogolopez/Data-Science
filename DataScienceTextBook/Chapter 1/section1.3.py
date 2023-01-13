import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [16,8]
plt.rcParams.update({'font.size':18})

theta = np.array([np.pi/15, -np.pi/9, -np.pi/20])
Signma = np.diag([3,1,0.5]) #Define Scaling Factors in X, Y, Z

#Define Rotations in X, Y, Z
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta[0]), -np.sin(theta[0])],
               [0, np.sin(theta[0]), np.cos(theta[0])]])
Ry = 