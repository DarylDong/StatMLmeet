import h5py
import matplotlib.pylab as plt
import numpy as np


hdf5_path = "../data/z0_100.hdf5"
hdf5_file = h5py.File(hdf5_path, mode='r')

data_sim = hdf5_file["sims_z0"]

print(data_sim.shape[0])

plt.figure(32) 
plt.imshow(data_sim[53, :, 60, :], cmap='gray')
plt.show() 
