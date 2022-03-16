% Function to create training data for the 1D wave equation 
% 
% Result is deflection over space and time y_x for differnt excitation
% positions with a delta-function at t=0. 
%
%
% M. Schäfer, 2022
%


addpath('./mat2npy');

%%%%%%%%%%%%%%%%%%%%%%%
% Basic Parameter Set %
%%%%%%%%%%%%%%%%%%%%%%%

f = 4410;      % sampling frequency  
T = 1/f;        % sampling time 
dur = 1;       % simulation duration

t = 0:T:dur;    % time vector 

l = 5;          % length of the pipe 
dx = 1e-2;      % spatial stepsize 
xs = 0:dx:l;    % space vector 

c0 = 340;       % propagation speed

%%%%%%%%%%%%%
% FTM Stuff %
%%%%%%%%%%%%%

Mu = 50;       % number of eigenvalues 
mu = 1:Mu;    

gmu = [mu*pi/l mu*pi/l]; 
smu = [1j*c0*mu*pi/l -1j*c0*mu*pi/l];

K1 = @(x) 1j*sin(gmu*x); 
K2 = @(x) 1j*smu.*sin(gmu*x); 
Ka1 = @(x) 1j/c0^2*conj(smu).*sin(gmu*x); 
Ka2 = @(x) 1j*sin(gmu*x); 

nmu = 1./(l/2*(c0^2*smu + conj(smu)));

A = diag(exp(smu*T)); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation for different excitation positions %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xeVec = [0.1*l, 0.2*l, 0.3*l]; % vector of excitaion positions (can be extended) 

for xe = 1:length(xeVec)
    
    % Excitation for the wave equation is a simple delta-impulse at
    % position xe
    % Possible extensions: 
    % - exciation by a hamming window to have a more smooth excitation 
    % - combination with a temporal exciation shape 
    yi = Ka2(xeVec(xe))*T; % set initial values for states
    
    % vectors 
    ybar = zeros(2*Mu, length(t)); 
    
    % set initial states
    ybar(:,1) = yi; 
    
    % processing to create time progression of individual states
    for k = 2:length(t)
        ybar(:,k) = A*ybar(:,k-1);
    end
    
    % create output signal over time at a single observation position
    % (maybe this part is not necessary, therefore it is commented)
    xo = 0.7*l; 
    c1 = K1(xo); 
    y = c1*ybar; % recover deflection from states (inverse transformation)
    y = real(y);
    
    
    % create spatial vectors. 
    % Result y_x: spatial distribution of the deflection y on the pipe at all
    % temportal sampling points
    
    K1_x = zeros(length(xs), 2*Mu); 
    y_x = zeros(length(xs), length(t)); 

    for xi = 1:length(xs) 
        K1_x(xi,:) = K1(xs(xi))./nmu; 
        y_x(xi,:) = K1_x(xi,:)*ybar; 
    end

    % take the real part because there might be a small imaginary part 
    y_x = real(y_x); 
    
    
    % save results into mat-files (maybe we need another format here, but I
    % don't know what is suitable for you to use in python? 
%     filename = sprintf('wq_yx_xe%d.mat', xeVec(xe));
%     save(filename,'y_x');
    
    % there is also some function to save for python? maybe helpful
%     filename = sprintf('wq_yx_xe%d.pkl', xeVec(xe));t%     mat2np(y_x, filename, 'double')
end






