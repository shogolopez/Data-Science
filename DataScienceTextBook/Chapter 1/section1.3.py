import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = [16,8]
plt.rcParams.update({'font.size':18})

theta = np.array([np.pi/15, -np.pi/9, -np.pi/20])
Sigma = np.diag([3,1,0.5]) #Define Scaling Factors in X, Y, Z

#Define Rotations in X, Y, Z
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta[0]), -np.sin(theta[0])],
               [0, np.sin(theta[0]), np.cos(theta[0])]])
Ry = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
               [0, 1, 0],
               [-np.sin(theta[1]), 0, np.cos(theta[1])]])
Rz = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
               [np.sin(theta[2]), np.cos(theta[2]), 0],
                [0, 0, 1]])

X = Rz @ Ry @ Rx @ Sigma

fig = plt.figure()
ax1 = fig.add_subplot(121, projection = '3d')
u = np.linspace(-np.pi, np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

surf1 =ax1.plot_surface(x,y,z,cmap='jet',alpha = 0.6)#,facecolors = plt.tricontour)
surf1.set_edgecolor('k')
ax1.set_xlim3d(-2,2)
ax1.set_ylim3d(-2,2)
ax1.set_zlim3d(-2,2)

xR = np.zeros_like(x)
yR = np.zeros_like(y)
zR = np.zeros_like(z)