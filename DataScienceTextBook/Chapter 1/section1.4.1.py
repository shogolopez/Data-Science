import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size':18})

x = 3 #The real slope for the data
a = np.arange(-2,2,0.01)
a = a.reshape(-1,1)

b = x*a + np.random.randn(*a.shape) #Noise

plt.plot(a, x*a, color='k', linewidth=2, label='True Line')
plt.plot(a,b,'x', color='r', markersize = 10, label='Noise Data')

U,S,VT = np.linalg.svd(a, full_matrices = False)
xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b #least squares fit
plt.plot(a, xtilde*a, '--', color='b', linewidth=4, label='Regression Line')

plt.xlabel('a')
plt.ylabel('b')

plt.legend()
plt.show()