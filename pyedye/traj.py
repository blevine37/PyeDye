import numpy as np
import h5py
import cmath

class traj(object):
    """A trajectory"""

    def __init__(self,numdims):
        self.time = 0.0
        self.numdims = numdims
        self.positions = np.zeros(self.numdims)
        self.momenta = np.zeros(self.numdims)
        self.s1hessian = np.zeros((self.numdims,self.numdims))
        self.s0hessian = np.zeros((self.numdims,self.numdims))
        self.s1gradient = np.zeros(self.numdims)
        self.s1energy = 0.0
        self.mfhessian = np.zeros((self.numdims,self.numdims))
        self.mfgradient = np.zeros(self.numdims)
        self.masses = np.ones(self.numdims)
        self.endtime = 0.0
        self.timestep = 0.0

        self.hamiltonian = np.zeros((2,2))
        self.propagator = np.zeros((2,2),dtype=complex)
        self.pe = 0.0
        self.ke = 0.0
        self.gradient = np.zeros(self.numdims)
        self.rho = np.zeros((2,2),dtype=complex)
        self.rho[1,1] = 1.0

    def compute_mfhessian(self):
        self.mfhessian = np.real(self.rho[0,0]) * self.s0hessian + np.real(self.rho[1,1]) * self.s1hessian
        self.mfgradient = np.real(self.rho[1,1]) * self.s1gradient
        
    def compute_gradient(self):
        self.gradient = 2.0 * np.dot(self.mfhessian,self.positions) + self.mfgradient

    def compute_pe(self):
        self.pe = np.real(np.sum(self.hamiltonian * self.rho))

    def compute_hamiltonian(self):
        self.hamiltonian = np.zeros((2,2))
        tmp = np.dot(self.s0hessian,self.positions)
        self.hamiltonian[0,0] = np.dot(self.positions,tmp)
        tmp = np.dot(self.s1hessian,self.positions)
        self.hamiltonian[1,1] = np.dot(self.positions,(tmp+self.s1gradient)) + self.s1energy

    def compute_propagator(self):
        self.compute_hamiltonian()
        [e,u]=np.linalg.eig(self.hamiltonian)
        tmp = np.zeros((2,2),dtype=complex)
        tmp[0,0] = cmath.exp(-0.5j*self.timestep*e[0])
        tmp[1,1] = cmath.exp(-0.5j*self.timestep*e[1])
        self.propagator = np.matmul(u,np.matmul(tmp,u.T))
        
    def compute_ke(self):
        inv2m = 0.5 / self.masses
        tmp = self.momenta * self.momenta
        tmp2 = tmp * inv2m
        self.ke = np.sum(tmp2)

    def compute_energy(self):
        self.compute_mfhessian()
        self.compute_hamiltonian()
        self.compute_pe()
        self.compute_ke()
        self.energy = self.ke + self.pe
        
    def propagate(self):
        hdt = 0.5 * self.timestep
        invm = 1.0 / self.masses
        self.compute_mfhessian()
        self.compute_gradient()
        self.compute_propagator()
        while self.time < self.endtime:
            self.momenta -= hdt * self.gradient

            tmp = np.matmul(self.rho, np.conjugate(self.propagator.T))
            self.rho = np.matmul(self.propagator, tmp)
            
            posdot = self.momenta * invm
            self.positions += self.timestep * posdot

            self.compute_mfhessian()
            self.compute_gradient()
            self.compute_propagator()

            tmp = np.matmul(self.rho, np.conjugate(self.propagator.T))
            self.rho = np.matmul(self.propagator, tmp)
            
            self.momenta -= hdt * self.gradient
            
            self.time += self.timestep
            
