import numpy as np
import h5py

class bundle(object):
    """A bundle of trajectories to be propagated simultaneously"""

    def __init__(self,numtraj,numdims):
        self.time = 0.0
        self.numdims = numdims
        self.numtraj = numtraj
        self.positions = np.zeros((self.numtraj,self.numdims))
        self.momenta = np.zeros((self.numtraj,self.numdims))
        self.hessian = np.zeros((self.numdims,self.numdims))
        self.masses = np.ones(self.numdims)
        self.endtime = 0.0
        self.timestep = 0.0
        
        self.energies = np.zeros(self.numtraj)
        self.PEs = np.zeros(self.numtraj)
        self.KEs = np.zeros(self.numtraj)
        
    def compute_PE(self):
        tmp = np.matmul(self.hessian,self.positions.T)
        print('tmp',tmp)
        tmp2 = self.positions * tmp.T
        print('tmp2',tmp2)
        self.PEs = np.sum(tmp2,axis=1)
        
