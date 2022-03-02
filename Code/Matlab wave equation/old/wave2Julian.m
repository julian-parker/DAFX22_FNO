%%%%%%%%%%%%%%%%%%%%%%%
% Basic Parameter Set %
%%%%%%%%%%%%%%%%%%%%%%%

f = 44100;      % sampling frequency  
T = 1/f;        % sampling time 
dur = 5;       % simulation duration

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


%%%%%%%%%%%%%%
% Excitation %
%%%%%%%%%%%%%%

xe = 0.5*l; 

y1 = 0; 
y2 = 1;

yi = Ka2(xe); 


%%%%%%%%%%%%%%
% Simulation %
%%%%%%%%%%%%%%

ybar = zeros(2*Mu,length(t)); 
y = zeros(1,length(t)); 

c1 = K1(0.8*l); 
A = diag(exp(smu*T)); 

ybar(:,1) = yi;

for k =2:length(t)
    
    ybar(:,k) = A*ybar(:,k-1);
    
end

y = c1*ybar;


%%%%%%%%%%%
% Spatial %
%%%%%%%%%%%

K1_x = zeros(length(xs), 2*Mu); 
y_x = zeros(length(xs), length(t)); 

for xi = 1:length(xs) 
   K1_x(xi,:) = K1(xs(xi))./nmu; 
   y_x(xi,:) = K1_x(xi,:)*ybar; 
end








