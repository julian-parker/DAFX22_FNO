import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import integrate
from numpy import sin, cos, conj, cumsum, real, zeros, pi
from numpy.random import rand

class StringSolver():
    def __init__(self):
        super(StringSolver, self).__init__()

        # Basic Parameter Set
        # Sampling frequency 
        Fs = 48000 
        T = 1/Fs
        
        # String parameters
        # Physical parameters for a nylon guitar B-string, see (Fletcher & Rossing
        # 1998), (Trautmann & Rabenstein, 2003)  
        E = 5.4e9
        p = 1140       
        l = 0.65
        A = 0.5188e-6  
        I = 0.171e-12
        d1 = 8 * 10**-5
        d3 = 1.4 * 10**-5
        Ts = 60.97     

        # Simulation domain
        delta_x = 1e-3
        numXs = round(l / delta_x)
        xs = np.linspace(0, l, num=numXs, endpoint=True) # space vector
        
        # Parameters for vector formulation
        c1 = -p*A/(E*I)
        c2 = d3/(E*I)
        a1 = d1/(E*I)
        a2 = -Ts/(E*I)
        
        ## FTM Parameters 
        # Number of modes used for synthesis (influences accuracy, should be adjusted
        # according to the hearing range)
        Nu = 50  # number of eigenvalues
        nu = np.arange(1, Nu+1) # 1:Nu 

        # Wavenumbers 
        gnu = nu*np.pi/l 
        gmu = np.concatenate((gnu, gnu)) # gmu = [gnu gnu]
        
        ## Eigenvalues 
        # Eigenvalues from dispersion relation 
        # Calculate corresponding frequency: imag(snu)/(2*pi)
        sigma = -0.5*(d3/(p*A)*gnu**2 + d1/(p*A))
        omega = np.sqrt((E*I)/(p*A)*gnu**4 + Ts/(p*A)*gnu**2 - sigma)

        snu = sigma + 1j*omega

        # Account for the complex conjugate pairs 
        smu = np.hstack((snu, conj(snu)))

        # Scaling factor 
        nmu = l/2*(2*smu*c1 - a1 - c2*gmu)

        # Eigenfunctions 
        self.K1 = lambda x: smu/gmu*sin(gmu*x)
        self.K2 = lambda x: cos(gmu*x)
        self.K3 = lambda x: -gmu*sin(gmu*x)
        self.K4 = lambda x: -gmu**2*cos(gmu*x) 

        self.Ka1 = lambda x: conj(smu*c1 - a1)*cos(gmu*x) 
        self.Ka2 = lambda x: -conj(smu)*conj(smu*c1 - a1)/gmu*sin(gmu*x) 
        self.Ka3 = lambda x: -gmu**2*cos(gmu*x) 
        self.Ka4 = lambda x: gmu*sin(gmu*x)

        
        # Explicity copy variables
        self.l, self.gmu, self.E, self.I, self.smu = l, gmu, E, I, smu
        self.nmu = nmu
        self.T = T
        self.Fs = Fs
        self.xs = xs
        
    def create_impulse(self, xe_rel):
        # Explicity copy variables
        l, gmu, Ka4, E, I = self.l, self.gmu, self.Ka4, self.E, self.I
        
        ## Input signal 

        # Use a delta at exciation position on the string
        xe = xe_rel*l
        x0 = 0.2

        fe_x = np.zeros((1, gmu.size)) # zeros(1,length(gmu))

        # impulse excitation at xe
        fe_x = -1/(E*I)*Ka4(xe) 

        return fe_x
    
    def create_pluck(self, xe_rel):
        # Explicity copy variables
        l, gmu, Ka4, E, I = self.l, self.gmu, self.Ka4, self.E, self.I
        
        ## Input signal 

        # Use a delta at exciation position on the string
        xe = xe_rel*l
        x0 = 0.2

        fe_x = np.zeros((1, gmu.size)) # zeros(1,length(gmu))

        # Hann window excitation centered at xe
        for mu in range(gmu.size) : # 1:length(gmu)
            g = gmu[mu]
            fun = lambda x: g*sin(g*x)*0.5*(1 + cos(2*pi/x0*(x-xe)))
            integ = integrate.quad(fun,xe-x0/2,xe+x0/2)
            fe_x[:,mu] = -1/(E*I)*integ[0]
            
        return fe_x

    def create_random_initial(self):
        # Explicity copy variables
        l, gmu, Ka4, E, I = self.l, self.gmu, self.Ka4, self.E, self.I
        
        fe_x = np.zeros((1, gmu.size)) # zeros(1,length(gmu))
        
        # Random excitation 
        for mu in range(gmu.size) :
            g = gmu[mu]
            r = (rand(1)*2-1)
            fun = lambda x: g*sin(g*x)*r
            integ = integrate.quad(fun,0,l)
            fe_x[:,mu] = -1/(E*I)*integ[0]
            
        return fe_x

    def solve(self, fe_x, dur):
        
        smu, nmu = self.smu, self.nmu
        T = self.T
        l = self.l
        K1 = self.K1
        xs = self.xs
        
        # Simulation duration 
        # dur = 2
        numT = round(dur / T)
        t = np.linspace(0, dur, num=numT, endpoint=True ) # time vector
        
        ## Simulation - state equation 
        ybar = np.zeros((smu.size, t.size),dtype=complex)

        # Matrix of eigenvalues, As: Frequency domain, Az: discrete-time domain
        As = np.diag(smu)
        Az = expm(As*T) 

        # Input at t = 0 --> is realized by an initial value for ybar 
        ybar[:,0] = fe_x

        for k in range(1,t.size) : # 1:length(t)
            # Process state equation 
            ybar[:,k] = Az@ybar[:,k-1]
        
        ## Simulation - Spatial domain 

        # create a spatial eigenfunction 
        K1_x = zeros((xs.size, smu.size),dtype=complex) 
        y_x = zeros((xs.size, t.size),dtype=complex)     
        y_defl_x = zeros((xs.size, t.size),dtype=complex) 

        for xi in range(xs.size) :
            K1_x[xi,:] = K1(xs[xi])/nmu 
            y_x[xi,:] = K1_x[xi,:]@ybar
            y_defl_x[xi,:] = cumsum(y_x[xi,:])*T
        
        return t, y_x, y_defl_x
        

