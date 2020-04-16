import numpy as np 
import h5py
from random import shuffle
import glob
import h5py 

ng = 64
zID = [0, 50]

for z in zID:
	hdf5_path = "../data/z"+str(z)+"_100.hdf5"
	sim_path = "/cosmo/scratch/nramachandra/Zeldo/z"+str(z)+"/fort.12*"
	addrs = glob.glob(sim_path)

	train_shape = (len(addrs), ng, ng, ng)
	hdf5_file = h5py.File(hdf5_path, mode='w')
	hdf5_file.create_dataset("rho", train_shape, np.float)

	for i in range(len(addrs)):
		#print('Data: {}/{}'.format(i, len(addrs))
		addr = addrs[i]
		rho1d = np.array(np.genfromtxt(addr, dtype = "f8"))
		rho3d = np.reshape(rho1d, (ng, ng, ng), order="F")
		hdf5_file["rho"][i, ...] = rho3d[None]

	hdf5_file.close()
