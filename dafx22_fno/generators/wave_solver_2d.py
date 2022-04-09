import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import integrate
from numpy import sin, cos, conj, cumsum, real, zeros, pi, trapz, arange, vstack, hstack, meshgrid, sqrt, diag, einsum, newaxis, float32
from numpy.random import rand

class WaveSolver2D():
    def __init__(self
                ,Fs = 48000 # sampling frequency 
                ,dur = 0.02 # duration of the simulation              
                ,lx = 1 # Room Parameters 
                ,ly = 1
                ,c0 = 340
                ,rho = 1.2041
                ,spatial_delta = 1e-3 # spatial grid
                ,damping = 1 # artificial damping factor. Maybe interesting to change during learning or estimation? 
                ,modes = 10
                ):
        super(WaveSolver2D, self).__init__()

        # Basic Parameter Set
        # Sampling frequency 
        
        T = 1/Fs

        Mux = modes## FTM - Parameters
        Muy = modes  

        self.numT = round(dur / T)
        self.t = np.linspace(0, dur, num=self.numT, endpoint=True ) # time vector

        self.numXs = round(lx / spatial_delta)
        self.numYs = round(ly / spatial_delta)
        xs = np.linspace(0, lx, num=self.numXs, endpoint=True) # space vector
        ys = np.linspace(0, ly, num=self.numYs, endpoint=True) # space vector
        

        # Create index vector. 
        # We have modes in 2 directions x,y but we want to count the eigenvalues
        # with one index 'mu' --> each 'mu' is a index tupel 
        # index = zeros((Mux*Muy,2))
        # index[:,2] = repmat(1:Muy,1,Mux)
        # index[:,1] = repelem(1:Mux, Muy)
        
        xv, yv = meshgrid(arange(Mux), arange(Muy))
        mux = xv.flatten() + 1
        muy = yv.flatten() + 1
        index = vstack((yv.flatten(),xv.flatten()))
        
        Mu = Mux*Muy
        mu = arange(Mu)
        smu = zeros((Mu),dtype=complex)

        ## FTM - Eigenvalues 
        # create wavenumbers 
        lamX = mux*pi/lx
        lamY = muy*pi/ly

        # create eigenvalues 
        smu[mu] = 1j*c0*sqrt(lamX[mu]**2 + lamY[mu]**2)

        # Now condsider complex conjugated eigenvalues 
        smu = hstack((smu, conj(smu)))
        lamX = hstack((lamX.transpose(), lamX.transpose()))
        lamY = hstack((lamY.transpose(), lamY.transpose()))
        Mu = smu.size

        # Add the artificial damping 
        smu = smu - damping 

        ## FTM - scaling factor 
        nx = -8*lamX**2/(rho*smu**2)*lx*ly
        ny = -8*lamY**2/(rho*smu**2)*lx*ly
        nc = 8/(rho*c0**2)*lx*ly

        nmu = nx+ny+nc

        ## FTM - Eigenfunctions 
        self.K1 = lambda x,y: 4*cos(lamX*x)*cos(lamY*y)
        self.K2 = lambda x,y: 4*lamX/(smu*rho)*sin(lamX*x)*cos(lamY*y)
        self.K3 = lambda x,y: 4*lamY/(smu*rho)*cos(lamX*x)*sin(lamY*y)

        self.Ka1 = lambda x,y: -4*lamX/(smu*rho)*sin(lamX*x)*cos(lamY*y)
        self.Ka2 = lambda x,y: -4*lamY/(smu*rho)*cos(lamX*x)*sin(lamY*y)
        self.Ka3 = lambda x,y: 4*cos(lamX*x)*cos(lamY*y)
        
        # Explicity copy variables
        self.lx = lx
        self.ly = ly        
        self.smu = smu
        self.nmu = nmu
        self.T = T
        self.Fs = Fs
        self.xs = xs
        self.ys = ys
        
        self.lamX = lamX
        self.lamY = lamY
        
    def create_impulse(self, xe_rel, ye_rel):
        # Explicity copy variables
        lx, ly = self.lx, self.ly
        smu = self.smu
        
        ## Input signal 

        # Use a delta at exciation position on the string 
        xe = xe_rel*lx
        ye = ye_rel*ly

        # impulse excitation at (xe,ye)
        fe_xy = self.Ka3(xe,ye)

        return fe_xy

    def create_random_initial(self):
        # Explicity copy variables
        lx, ly = self.lx, self.ly
        xs, ys = self.xs, self.ys
        smu = self.smu
        lamX, lamY = self.lamX, self.lamY
        
        fe_xy = np.zeros((1, smu.size))
        
        # Random excitation
        for mu in range(smu.size) :
            g = smu[mu]
            rx = (rand(1,xs.size)*2-1)
            ry = (rand(1,ys.size)*2-1)
            
            Kx = 4*cos(lamX[mu]*xs) 
            Ky = cos(lamY[mu]*ys)
            
            funX = 4*cos(lamX[mu]*xs)*(rx)
            funY = cos(lamY[mu]*ys)*(ry)
            
            integ = trapz(xs,funX)*trapz(ys,funY)
            fe_xy[:,mu] = integ[0]
            
            
        return fe_xy
        

    def solve(self, fe_x):
        ## Copy internal variables
        T = self.T
        smu, nmu = self.smu, self.nmu
        lx, ly = self.lx, self.ly
        xs, ys = self.xs, self.ys
    
        ## Simulation - state equation 

        ybar = zeros((smu.size, self.t.size),dtype=complex)

        # Matrix of eigenvalues, As: Frequency domain, Az: discrete-time domain
        As = diag(smu)
        Az = expm(As*T) 

        # Input at t = 0 --> is realized by an initial value for ybar 
        ybar[:,0] = fe_x

        for k in range(1,self.t.size) : #for k = 2:length(t) 
            # Process state equation 
            ybar[:,k] = Az@ybar[:,k-1]
        

        ## Simulation - output equation 
        xo = 0.7*lx 
        yo = 0.7*ly 

        # take first entry (sound pressure) 
        y = (self.K1(xo,yo)/nmu)@ybar
        y = real(y)
        
        ## Simulation - spatial domain 
        K1_sp = zeros((xs.size, ys.size, nmu.size),dtype=complex) # Eigenfunctions for sound pressure 
        K2_vx = zeros((xs.size, ys.size, nmu.size),dtype=complex) # Eigenfunctions for particle velocity in x-direction
        K3_vy = zeros((xs.size, ys.size, nmu.size),dtype=complex) # Eigenfunctions for particle velocity in y-direction 
        
        y_sp = zeros((xs.size, ys.size, self.t.size),dtype=complex) # Output sound pressure
        y_vx = zeros((xs.size, ys.size, self.t.size),dtype=complex) # output particle velocity in x-direction
        y_vy = zeros((xs.size, ys.size, self.t.size),dtype=complex) # output particle velocity in y-direction
        
        for xi in range(xs.size) :
            for yi in range(ys.size) :
                K1_sp[xi,yi,:] = self.K1(xs[xi], ys[yi])/nmu
                K2_vx[xi,yi,:] = self.K2(xs[xi], ys[yi])/nmu
                K3_vy[xi,yi,:] = self.K3(xs[xi], ys[yi])/nmu
                
                y_sp[xi,yi,:] = K1_sp[xi,yi,:]@ybar
                y_vx[xi,yi,:] = K2_vx[xi,yi,:]@ybar
                y_vy[xi,yi,:] = K3_vy[xi,yi,:]@ybar
        
        y = float32(real(y))
        ybar = float32(real(ybar))
        y_sp = float32(real(y_sp))
        y_vx = float32(real(y_vx))
        y_vy = float32(real(y_vy))

        return y, ybar, y_sp, y_vx, y_vy
    
    
    
