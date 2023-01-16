import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size':18})

#Importing Data
A = np.loadtxt(os.path.join('hald_ingredients.csv'),delimiter=',')
B = np.loadtxt(os.path.join('hald_heat.csv'),delimiter=',')

U, S, VT = np.linalg.svd(A, full_matrices=0)

x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ B

plt.plot(B, color='k', linewidth =2, label = "Heat Data")
plt.plot(A @ x, '-o', color='r', linewidth=1.5, markersize=6, label='Regression')
plt.legend()
plt.show()