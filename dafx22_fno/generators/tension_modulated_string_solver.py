import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import integrate
from numpy import sin, cos, conj, cumsum, real, zeros, pi, trapz, arange, vstack, hstack, meshgrid, sqrt, diag, einsum, newaxis, float32
from numpy.random import rand
from scipy.integrate import solve_ivp

class TensionModulatedStringSolver():
    def __init__(self
                ,Fs  = 48000        # 1/s           Temporal sampling frequency
                ,dur = 0.1          # s             duration of the simulation
                ,ell = 1            # m             string length at rest
                ,A   = 0.19634e-6   # m**2           string cross section area
                ,I   = 0.02454e-12  # m**4           string moment of intertia
                ,rho = 7800         # kg/m**3        string density
                ,E   = 190e9        # Pa            string elasticity
                ,d1  = 4e-3         # kg/(ms)       string frequ. independent loss
                ,d3  = 6e-5         # kg m/s        string frequ. dependent loss
                ,Ts0 = 150          # N             string tension
                ,M   = 50           #               number of expansion terms  
                ):
        super(TensionModulatedStringSolver, self).__init__()
        
        mu           = np.arange(1,M+1)                 # index for Fourier-Sine transf.
        self.kmu     = mu*pi/ell              # argument of sine functions

        self.Fs = Fs
        self.dur = dur
        self.ell = ell
        self.A = A
        self.I = I
        self.rho = rho
        self.E = E
        self.d1 = d1
        self.d3 = d3
        self.Ts0 = Ts0
        self.M = M       
        
    def create_pluck(self, xe_rel, hi):
        kmu = self.kmu
        
        xe  = self.ell * xe_rel  # m          pluck position
        
        # Fourier-Sine coefficients of the initial deflection
        yb0     = hi *(self.ell/(self.ell-xe)*sin(kmu*xe)/(kmu*xe))/kmu

        zb0     = zeros((self.M))               # initial condition for derivative 
        wb0     = np.concatenate((yb0, zb0))    # initial condition for ode   

        
        return wb0

    def create_random_initial(self):
        
            
        return fe_xy
        

    def tensmodstr(self,t,wb):

        # extract parameter values from self
        ell     = self.ell
        E       = self.E
        I       = self.I
        rho     = self.rho
        A       = self.A
        d1      = self.d1
        d3      = self.d3
        Ts0     = self.Ts0
        M       = self.M

        # extract yb and zb from wb
        yb      = wb[:M]
        zb      = wb[M:2*M]

        # set up vectors and matrices
        mu      = np.arange(1,M+1)
        kmu     = mu*pi/ell
        Mz      = diag((d1+d3*kmu**2)/(rho*A))
        My      = diag((Ts0*kmu**2 + E*I*kmu**4)/(rho*A))
        M1      = diag(kmu**2/(rho*A))
        M2      = diag(mu**2)

        # calculate additional string tension Ts1
        Ts1     = E*A*pi**2/ell**4 * yb[newaxis,:]@M2@yb 

        # calculate first order derivatives

        dyb_dt  = zb
        dzb_dt  = - Mz@zb - My@yb      # linear terms
        dzb_dt  = dzb_dt - Ts1*M1@yb   # nonlinear term
        dwb_dt  = np.concatenate((dyb_dt, dzb_dt))
        
        return dwb_dt
        

    def solve(self, wb0):
        ## Copy internal variables
        kmu = self.kmu
        
        # xa  = 0.28         # m             listening position
        xa  = np.linspace(0, self.ell, num=100) # 100 positions on the string
        xa  = xa[newaxis,:]

        tspan   = [0, self.dur]                   # time span for ode
        t_eval = np.linspace(0, self.dur, round(self.dur * self.Fs))
        
        #[t1,wb1] = ode45(@(t1,wb1) tensmodstr(t1,wb1,P),tspan,wb0) # solve ode
        vdp1 = lambda t1, wb1: self.tensmodstr(t1,wb1)
        sol = solve_ivp (vdp1, tspan, wb0, t_eval = t_eval)
        t1 = sol.t
        wb1 = sol.y

        yb1 = wb1[:self.M,:].transpose()   # solution for the FS-transform of the deflection

        # inverse Fourier-Sine transformation
        y1      = yb1@sin(kmu[:,newaxis]*xa) * 2/self.ell

        return t1, y1