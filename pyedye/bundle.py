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
        self.gradients = np.zeros((self.numtraj,self.numdims))

    def compute_gradients(self):
        self.gradients = 2.0*np.matmul(self.positions,self.hessian)
        
    def compute_PE(self):
        tmp = np.matmul(self.positions,self.hessian)
        tmp2 = self.positions * tmp
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

    def propagate(self):
        while self.time < self.endtime:
            posdot = np.zeros((self.numtraj,self.numdims))
            invm = 1.0 / self.masses
            hdt = 0.5 * self.timestep
            
            for i in range(self.numtraj):
                posdot[i,:] = self.momenta[i,:] * invm

            self.positions += hdt * posdot

            self.compute_gradients()

            self.momenta -= self.timestep * self.gradients

            for i in range(self.numtraj):
                posdot[i,:] = self.momenta[i,:] * invm

            self.positions += hdt * posdot

            self.time += self.timestep
        
    
