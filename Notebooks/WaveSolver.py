import numpy as np
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class WaveSolver():
    def __init__(self):
        super(WaveSolver, self).__init__()

        # Basic Parameter Set
        self.f = 4800      # sampling frequency
        self.T = 1/self.f    # sampling time
        self.dur = 0.01       # simulation duration

        self.numT = round(self.dur / self.T)
        self.t = np.linspace(0, self.dur, num=self.numT, endpoint=True ) # time vector

        self.numXs = 256      # spatial grid points
        self.l = 2.5          # length of the pipe
        self.xs = np.linspace(0.0, self.l, num=self.numXs, endpoint=True) # space vector

        self.c0 = 30       # propagation speed

    def create_pluck(self, xe):
        #xeVec = np.array([0.1*self.l, 0.2*self.l, 0.3*self.l]) # vector of excitation positions (can be extended)
        # xe = 0.1*l;
        xi = find_nearest(self.xs, xe * self.l)
        initial_condition = np.zeros((self.numXs, 1))
        # initial_condition[xi] = 1;
        xi = np.minimum(self.numXs-12,xi)
        initial_condition[xi:12+xi,0] = np.hanning(12)
        return initial_condition

    def create_random_initial(self):
        order = 6
        fs = 4410
        cutoff = 300
        initial_condition = np.random.randn(self.numXs, 1) * 10
        initial_condition[:20] = 0
        initial_condition[-20:] = 0
        initial_condition = butter_lowpass_filter(initial_condition.transpose(), cutoff, fs, order).transpose()

        return initial_condition


    def solve(self, initial_condition):
        # FTM Stuff
        Mu = 250       # number of eigenvalues
        mu = np.arange(1, Mu+1) # 1:Mu;
        c0 = self.c0
        l = self.l
        numT = self.numT
        numXs = self.numXs

        test = 1j*c0*mu*np.pi/l

        gmu = np.concatenate((mu*np.pi/l, mu*np.pi/l))
        smu = np.concatenate((1j*c0*mu*np.pi/l, -1j*c0*mu*np.pi/l))

        K1 = lambda x: 1j*np.sin(gmu*x) # @(x) 1j*sin(gmu*x);
        K2 = lambda x: 1j*smu*np.sin(gmu*x)
        Ka1 = lambda x: 1j/c0**2*np.conj(smu)*np.sin(gmu*x)
        Ka2 = lambda x: 1j*np.sin(gmu*x)

        nmu = 1./(l/2*(c0**2*smu + np.conj(smu)))

        A = np.diag(np.exp(smu*self.T))



        # Excitation for the wave equation is a simple delta-impulse at position xe
        # Possible extensions:
        # - excitation by a hamming window to have a more smooth excitation
        # - combination with a temporal excitation shape

        # initial_condition = initial_condition / sum(initial_condition**2) * self.T
        yi = np.zeros(2*Mu,dtype=complex)
        for xi in range(numXs) : #1:length(xs)
            yi += Ka2(self.xs[xi]) * self.T * initial_condition[xi]

        #yi = Ka2(xe)*self.T # set initial values for states

        # vectors
        ybar = np.zeros((2*Mu, numT),dtype=complex)

        # set initial states
        ybar[:,0] = yi

        test = range(1,numT)

        # processing to create time progression of individual states
        for k in range(1,numT) :
            ybar[:,k] = A@ybar[:,k-1]

        # create output signal over time at a single observation position
        # (maybe this part is not necessary, therefore it is commented)
        # xo = 0.7*l
        # c1 = K1(xo)
        # y = c1@ybar # recover deflection from states (inverse transformation)
        # y = np.real(y)

        # create spatial vectors.
        # Result y_x: spatial distribution of the deflection y on the pipe at all
        # temporal sampling points
        K1_x = np.zeros((numXs, 2*Mu),dtype=complex)
        y_x = np.zeros((numXs, numT),dtype=complex)

        for xi in range(numXs) : #1:length(xs)
            K1_x[xi,:] = K1(self.xs[xi])/nmu
            y_x[xi,:] = K1_x[xi,:]@ybar

        # take the real part because there might be a small imaginary part
        y_x = np.real(y_x)
        # y_x = y_x / 10**6 # scale the output to reasonable values around 1
        y_x = y_x / y_x.std();

        return y_x