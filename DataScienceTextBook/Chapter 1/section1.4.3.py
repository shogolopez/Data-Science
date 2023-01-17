import numpy as np
import matplotlib.pyplot as plt
import os
plt.rcParams['figure.figsize'] = [8,8]
plt.rcParams.update({'font.size':18})

#Loading data...
H = np.loadtxt(os.path.join('housing.data'))
b = H[:,-1]
A = H[:,:-1]

A = np.pad(A,[(0,0),(0,1)],mode='constant',constant_values=5)

U, S, VT = np.linalg.svd(A, full_matrices = 0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.plot(b, color='k', linewidth=2, label="Housing Value")
plt.plot(A@x, '-o', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value')

ax2 = fig.add_subplot(122)
sort_ind = np.argsort(H[:,-1])
b = b[sort_ind]
plt.plot(b, color='k', linewidth=2, label="Housing Value")
plt.plot(A[sort_ind,:]@x, '-o', linewidth=1.5, markersize=6, label='Regression')
plt.legend()
plt.show()

#Significance of Various Attributes
A_mean = np.mean(A,axis=0)
A_mean = A_mean.reshape(-1,1)

A2 = A-np.ones((A.shape[0],1)) @ A_mean.T

for j in range(A.shape[1]-1):
    A2std = np.std(A2[:,j])
    A2[:,j] = A2[:,j]/A2std

A2[:,-1] = np.ones(A.shape[0])

U, S, VT = np.linalg.svd(A2, full_matrices=0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
x_tick = range(len(x)-1)+np.ones(len(x)-1)
plt.bar(x_tick,x[:-1])
plt.xlabel('Attribute')
plt.ylabel('Significance')
plt.xticks(x_tick)
plt.show()
##Reloading data and instead Training on half and testing on other half
H = np.loadtxt(os.path.join('housing.data'))
b = H[:,-1]
A = H[:,:-1]

A = np.pad(A,[(0,0),(0,1)],mode='constant',constant_values=1)

n = int(len(H)/2)
btrain = b[1:n]
Atrain = A[1:n]
btest = b[n:]
Atest = A[n:]

U, S, VT = np.linalg.svd(Atrain, full_matrices = 0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ btrain
fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.plot(btrain, color='k', linewidth=2, label="Housing Value")
plt.plot(Atrain@x, '-o', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value')

ax2 = fig.add_subplot(122)
plt.plot(btest, color='k', linewidth=2, label="Housing Value")
plt.plot(Atest@x, '-o', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value')
plt.legend()
plt.show()

##Reloading data and instead randomly shuffle rows of A and B the same way, and train and test the data
H = np.loadtxt(os.path.join('housing.data'))
b = H[:,-1]
A = H[:,:-1]

A = np.pad(A,[(0,0),(0,1)],mode='constant',constant_values=1)

n = int(len(H)/2)
p = np.random.permutation(int(len(H)))
A = A[p,:]
b = b[p]
btrain = b[1:n]
Atrain = A[1:n]
btest = b[n:]
Atest = A[n:]

U, S, VT = np.linalg.svd(Atrain, full_matrices = 0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ btrain
fig = plt.figure()
ax1 = fig.add_subplot(121)
plt.plot(btrain, color='k', linewidth=2, label="Housing Value")
plt.plot(Atrain@x, '-o', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value')

ax2 = fig.add_subplot(122)
plt.plot(btest, color='k', linewidth=2, label="Housing Value")
plt.plot(Atest@x, '-o', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value')
plt.legend()
plt.show()