from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

A = imread(os.path.join("C:\\Users\\shogo\\OneDrive\\Pictures\\Camera Roll\\unnamed.jpg"))
X = np.mean(A, -1)

##PLOTTING
# Adds a subplot at the 1st position
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 2

fig.add_subplot(rows, columns, 1)
img = plt.imshow(X)
img.set_cmap('gray')
# showing image
#fig.add_subplot(rows, columns, 1)
#img = plt.imshow(X)
#img.set_cmap('gray')
#plt.show()

U, s, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(s)
j=1
for r in (5, 20, 100):
    j=j+1
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]

    fig.add_subplot(rows, columns, j)
    plt.imshow(Xapprox)
    plt.set_cmap("gray")



plt.figure(2)
plt.semilogy(np.diag(S))
plt.title('Singlar Values')


plt.figure(3)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singlar Values: Cumulative Sum')
plt.show()