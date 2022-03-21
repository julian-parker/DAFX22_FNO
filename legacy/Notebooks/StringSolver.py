import numpy as np
from scipy.signal import butter, lfilter, freqz


class StringSolver():
    def __init__(self):
        super(StringSolver, self).__init__()

        # Basic Parameter Set
        # Sampling frequency 
        self.Fs = 48000 
        self.T = 1/self.Fs
        
        # Simulation duration 
        self.dur = 2;
        self.t = np.linspace(0, self.dur, num=self.numT, endpoint=True ) # time vector

        # String parameters
        # Physical parameters for a nylon guitar B-string, see (Fletcher & Rossing
        # 1998), (Trautmann & Rabenstein, 2003)  
        self.E = 5.4e9;
        self.p = 1140;       
        self.l = 0.65;
        self.A = 0.5188e-6;  
        self.I = 0.171e-12;
        self.d1 = 8 * 10^-5;
        self.d3 = 1.4 * 10^-5;
        self.Ts = 60.97;     

        # Parameters for vector formulation
        self.c1 = -self.p*self.A/(self.E*self.I);
        self.c2 = self.d3/(self.E*self.I);
        self.a1 = self.d1/(self.E*self.I);
        self.a2 = -self.Ts/(self.E*self.I);
        
        

    def create_pluck(self, xe):
        ## Input signal 

        # Use a delta at exciation position on the string 
        xe = 0.2*l;
        x0 = 0.2;

        fe_x = zeros(1,length(gmu));

        # impulse excitation at xe
        # fe_x = -1/(E*I)*Ka4(xe); 

        # Hann window excitation centered at xe
        for mu = 1:length(gmu)
            g = gmu(mu);
            fun = @(x) g.*sin(g.*x)*0.5.*(1 + cos(2*pi/x0.*(x-xe)));
            fe_x(mu) = -1/(E*I)*integral(fun,xe-x0/2,xe+x0/2); 
        end
   

    def create_random_initial(self):
        # Random excitation 
        # for mu = 1:length(gmu)
        #     g = gmu(mu); 
        #     fun = @(x) g.*sin(g.*x)*(rand*2-1);
        #     fe_x(mu) = -1/(E*I)*integral(fun,0,l); 
        # end 

    def solve(self, initial_condition):
        ## FTM Parameters 
        # Number of modes used for synthesis (influences accuracy, should be adjusted
        # according to the hearing range)
        Nu = 50;  # number of eigenvalues
        nu = np.arange(1, Nu+1) # 1:Nu; 

        # Wavenumbers 
        gnu = nu*pi/l; 
        gmu = np.concatenate((gnu, gnu)) # gmu = [gnu gnu];
        
        ## Eigenvalues 
        # Eigenvalues from dispersion relation 
        # Calculate corresponding frequency: imag(snu)/(2*pi)
        sigma = -0.5*(d3/(p*A)*gnu.^2 + d1/(p*A));
        omega = sqrt((E*I)/(p*A)*gnu.^4 + Ts/(p*A)*gnu.^2 - sigma);

        snu = sigma + 1j*omega;

        # Account for the complex conjugate pairs 
        smu = [snu conj(snu)];

        # Scaling factor 
        nmu = l/2*(2*smu*c1 - a1 - c2*gmu);

        # Eigenfunctions 
        K1 = @(x) smu./gmu.*sin(gmu*x);
        K2 = @(x) cos(gmu*x);
        K3 = @(x) -gmu.*sin(gmu*x);
        K4 = @(x) -gmu.^2.*cos(gmu*x); 

        Ka1 = @(x) conj(smu*c1 - a1).*cos(gmu*x); 
        Ka2 = @(x) -conj(smu).*conj(smu*c1 - a1)./gmu.*sin(gmu*x); 
        Ka3 = @(x) -gmu.^2.*cos(gmu*x); 
        Ka4 = @(x) gmu.*sin(gmu*x);



        ## Simulation - state equation 
        ybar = zeros(length(smu), length(t));

        # Matrix of eigenvalues, As: Frequency domain, Az: discrete-time domain
        As = diag(smu);
        Az = expm(As*T); 

        # Input at t = 0 --> is realized by an initial value for ybar 
        ybar(:,1) = fe_x;

        for k = 2:length(t) 
            # Process state equation 
            ybar(:,k) = Az*ybar(:,k-1);
        end

        ## Simulation - Output equation 
        # Set observation point 
        xo = 0.8*l;

        # Calculate output 
        # Use K1 - K4 depending on the desired physical quantity 
        # Here: K1 delivers the velocity (d/dt y) 
        y = (K1(xo)./nmu)*ybar;

        # Convert into deflection by integration 
        y_defl = cumsum(y)*T;

        # plot time signal
        # figure(2);  
        # plot(t,real(y_defl)); grid on;
        # ylabel('$y^(x_\mathrm{o},t)$', 'fontsize',14,'interpreter','latex'); 
        # xlabel('Time $t$ in $[\mathrm{s}]$', 'fontsize',14,'interpreter','latex');

        y = y_defl;

        # Sound output
        y = real(y); 
        y = y/max(y);
        sound(y,Fs);


        ## Simulation - Spatial domain 

        # create a spatial eigenfunction 
        delta_x = 1e-3;
        xs = 0:1e-3:l;

        K1_x = zeros(length(xs), 2*Nu); 
        y_x = zeros(length(xs), length(t));     
        #y_defl_x = zeros(length(xs), length(t)); 

        for xi = 1:length(xs) 
          K1_x(xi,:) = K1(xs(xi))./nmu; 
          y_x(xi,:) = K1_x(xi,:)*ybar;
          #y_defl_x(xi,:) = cumsum(y_x(xi,:))*T;
        end