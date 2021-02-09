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
        tmp2 = self.positions * tmp.T
        self.PEs = np.sum(tmp2,axis=1)

    def compute_KE(self):
        tmp = self.momenta * self.momenta
        inv2m = 1.0 / (2.0 * self.masses)
        for i in range(self.numtraj):
            tmp2 = tmp[i,:] * inv2m
            self.KEs[i] = np.sum(tmp2)
            
    def compute_energies(self):
        self.compute_KE()
        self.compute_PE()
        self.energies = self.PEs + self.KEs

    
