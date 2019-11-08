import numpy as np
import matplotlib.pylab as plt

dirIn = "/cosmo/scratch/nramachandra/Zeldo/"
zIn = [0, 50][1]
ng = 64


simID = np.arange(423, 523)[22]

#for simID in simIDall:
fileIn = dirIn + 'z' + str(zIn) + '/fort.1298' + str(simID)


rho1d = np.array(np.genfromtxt(fileIn, dtype = "f8"))

rho3d = np.reshape(rho1d, (ng, ng, ng), order="F")
plt.figure(32)
plt.imshow(rho3d[23, :, :] > 0)

plt.show()
