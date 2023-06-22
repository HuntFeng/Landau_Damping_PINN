# A quick introduction and implementation
# http://hsxie.me/codes/landaudamping/landaudamping_hsxie_beamer.pdf

import numpy as np

class Solver:
    def __init__(self, x: np.array, v: np.array, t:np.array, f:np.array) -> None:
        # params
        self.qi = 1.0
        self.qe = -1.0
        
        # mesh
        self.x, self.v, self.t = x, v, t
        nx, nv, nt = x.size, v.size, t.size
        self.dx = dx = x[1]-x[0]
        self.dv = dv = v[1]-v[0]
        self.dt = dt = t[1]-t[0]

        # solution
        self.f = f

        # Laplacian operator
        Laplacian = np.diag(np.ones(nx-1), k=-1) + np.diag(np.ones(nx-1), k=1) - 2* np.diag(np.ones(nx), k=0)
        Laplacian[0,-1] = 1
        Laplacian[-1,0] = 1
        Laplacian /= dx**2
        self.Laplacian = Laplacian

        # fields
        X,T = np.meshgrid(x,t)
        self.X, self.T = X, T
        self.n_tx = np.zeros_like(X)
        self.u_tx = np.zeros_like(X)
        self.p_tx = np.zeros_like(X)
        self.q_tx = np.zeros_like(X)
        self.E_tx = np.zeros_like(X)

        self.frame = 0

    def compute_fields(self, f:np.array, save=False):
        """ Compute all fields """
        x, v = self.x, self.v
        dx, dv = self.dx, self.dv

        n = f.sum(axis=1)*dv
        v_avg = (np.tile(v,(f.shape[0],1)) * f).sum(axis=1)*dv
        v_sqr_avg = (np.tile(v**2,(f.shape[0],1)) * f).sum(axis=1)*dv
        v_cube_avg = (np.tile(v**3,(f.shape[0],1)) * f).sum(axis=1)*dv
        u = v_avg / n
        p = v_sqr_avg - u**2
        q = v_cube_avg - 3*v_sqr_avg*u + 3*v_avg*u**2 - u**3
        
        rho = self.qi*1.0 + self.qe*n # charge density
        phi = np.linalg.solve(self.Laplacian, -rho) # poisson equations
        E = -(np.roll(phi,-1,axis=0) - np.roll(phi,1,axis=0))/(2*dx) 

        if save:
            self.n_tx[self.frame,:] = n
            self.u_tx[self.frame,:] = u
            self.p_tx[self.frame,:] = p
            self.q_tx[self.frame,:] = q
            self.E_tx[self.frame,:] = E

        return E # rhs needs the electric


    def rhs(self, f: np.array, E: np.array) -> np.array:
        """ df/dt = rhs(f,t) """
        dx, dv = self.dx, self.dv
        
        df_dx = (np.roll(f,-1,axis=0) - np.roll(f,1,axis=0))/(2*dx)
        df_dv = (np.roll(f,-1,axis=1) - np.roll(f,1,axis=1))/(2*dv) # don't worry the boundary because they are close to 0

        v_stack = np.tile(v, (df_dx.shape[0],1))
        E_stack = np.column_stack([E for _ in range(df_dv.shape[1])])

        return -v_stack*df_dx - self.qe*E_stack*df_dv

    def rk2(self, f:np.array, dt:float) -> np.array:
        """ RK2 time integration """
        E = self.compute_fields(f, save=True)
        k1 = dt*self.rhs(f,E)
        E_half = self.compute_fields(f+k1/2)
        k2 = dt*self.rhs(f+k1/2, E_half)
        return f + k2

    def run(self, filename:str):
        """ Run the simulation and save the data """
        f = self.f
        for frame in tqdm(range(self.t.size)):
            self.frame = frame
            f = solver.rk2(f,dt)
        solver.rk2(f,dt) # compute one more time to save last fields
        np.savez(
            filename, 
            T=self.T, 
            X=self.X, 
            n=self.n_tx, 
            u=self.u_tx, 
            p=self.p_tx, 
            q=self.q_tx, 
            E=self.E_tx
            )


if __name__ == '__main__':
    from scipy.stats import norm
    from tqdm import tqdm
    import os
    import shutil

    datadir = "simulation"
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
    os.mkdir(datadir)

    # parameters
    k1 = 0.6
    k2 = 1.2
    A1 = 0.05
    A2 = 0.4
    phase = 0.38716
    L = 2*np.pi/k1
    v_sigma = 1
    n0 = 1

    # mesh
    nx, nv = 128, 128
    x = np.linspace(0,L,nx,endpoint=False)
    v = np.linspace(-6*v_sigma,6*v_sigma,nv)
    tf = 10
    dt = 0.001
    t = np.arange(0,tf+dt,dt)

    # initial condition
    fv,fx = np.meshgrid(norm.pdf(v,0,v_sigma), n0*(1 + A1*np.cos(k1*x) + A2*np.cos(k2*x+phase)))
    f = fx*fv # f[xi,vj]

    # initialize solver
    solver = Solver(x,v,t, f)
    solver.run(f"{datadir}/data.npz")

