%% Physical Model of a guitar string 
%
%   Synthesis is based on the Functional Transformation Method (FTM)
% 
%   Dr. Maximilian Schaefer @FAU, 2021
%   Contact: max.schaefer@fau.de
%
%   References: see lecture slides 

%% Basic simulation param 
% Sampling frequency 
Fs = 48000; 
T = 1/Fs;
% Simulation duration 
dur = 2;
t = 0:T:dur-T;

%% String parameters
% Physical parameters for a nylon guitar B-string, see (Fletcher & Rossing
% 1998), (Trautmann & Rabenstein, 2003)  
E = 5.4e9;
p = 1140;       
l = 0.65;
A = 0.5188e-6;  
I = 0.171e-12;
d1 = 8 * 10^-5;
d3 = 1.4 * 10^-5;
Ts = 60.97;     

% Parameters for vector formulation
c1 = -p*A/(E*I);
c2 = d3/(E*I);
a1 = d1/(E*I);
a2 = -Ts/(E*I);


%% FTM Parameters 
% Number of modes used for synthesis (influences accuracy, should be adjusted
% according to the hearing range)
Nu = 50; 
nu = 1:Nu; 

% Wavenumbers 
gnu = nu*pi/l; 
gmu = [gnu gnu];

%% Eigenvalues 

% Eigenvalues from dispersion relation 
% Calculate corresponding frequency: imag(snu)/(2*pi)
sigma = -0.5*(d3/(p*A)*gnu.^2 + d1/(p*A));
omega = sqrt(...
    (E*I)/(p*A)*gnu.^4 + Ts/(p*A)*gnu.^2 - sigma);

snu = sigma + 1j*omega;

% Account for the complex conjugate pairs 
smu = [snu conj(snu)];

%% Scaling factor 
nmu = l/2*(2*smu*c1 - a1 - c2*gmu);

%% Eigenfunctions 

K1 = @(x) smu./gmu.*sin(gmu*x);
K2 = @(x) cos(gmu*x);
K3 = @(x) -gmu.*sin(gmu*x);
K4 = @(x) -gmu.^2.*cos(gmu*x); 

Ka1 = @(x) conj(smu*c1 - a1).*cos(gmu*x); 
Ka2 = @(x) -conj(smu).*conj(smu*c1 - a1)./gmu.*sin(gmu*x); 
Ka3 = @(x) -gmu.^2.*cos(gmu*x); 
Ka4 = @(x) gmu.*sin(gmu*x);

%% Input signal 

% Use a delta at exciation position on the string 
xe = 0.2*l;
x0 = 0.2;

fe_x = zeros(1,length(gmu));

% impulse excitation at xe
% fe_x = -1/(E*I)*Ka4(xe); 

% Hann window excitation centered at xe
% for mu = 1:length(gmu)
%     g = gmu(mu);
%     fun = @(x) g.*sin(g.*x)*0.5.*(1 + cos(2*pi/x0.*(x-xe)));
%     fe_x(mu) = -1/(E*I)*integral(fun,xe-x0/2,xe+x0/2); 
% end

% Random excitation 
for mu = 1:length(gmu)
    g = gmu(mu); 
    fun = g.*sin(g.*x).*(rand(1,length(x))*2-1);
    fe_x(mu) = -1/(E*I)*trapz(x,fun);
    %fun = @(x) g.*sin(g.*x)*(rand*2-1);
    %fe_x(mu) = -1/(E*I)*integral(fun,0,l); 
end    

%% Simulation - state equation 

ybar = zeros(length(smu), length(t));

% Matrix of eigenvalues, As: Frequency domain, Az: discrete-time domain
As = diag(smu);
Az = expm(As*T); 

% Input at t = 0 --> is realized by an initial value for ybar 
ybar(:,1) = fe_x;

for k = 2:length(t) 
    % Process state equation 
    ybar(:,k) = Az*ybar(:,k-1);
end

%% Simulation - Output equation 
% Set observation point 
xo = 0.8*l;

% Calculate output 
% Use K1 - K4 depending on the desired physical quantity 
% Here: K1 delivers the velocity (d/dt y) 
y = (K1(xo)./nmu)*ybar;

% Convert into deflection by integration 
y_defl = cumsum(y)*T;

% plot time signal
% figure(2);  
% plot(t,real(y_defl)); grid on;
% ylabel('$y^(x_\mathrm{o},t)$', 'fontsize',14,'interpreter','latex'); 
% xlabel('Time $t$ in $[\mathrm{s}]$', 'fontsize',14,'interpreter','latex');

y = y_defl;

% Sound output
y = real(y); 
y = y/max(y);
% sound(y,Fs);


%% Simulation - Spatial domain 

% create a spatial eigenfunction 
delta_x = 1e-3;
xs = 0:1e-3:l;

K1_x = zeros(length(xs), 2*Nu); 
K2_x = zeros(length(xs), 2*Nu); 
y_x = zeros(length(xs), length(t));
y_defl_x = zeros(length(xs), length(t)); 

for xi = 1:length(xs) 
  K1_x(xi,:) = K1(xs(xi))./nmu; 
  K2_x(xi,:) = K2(xs(xi))./nmu; 
  y_x(xi,:) = K1_x(xi,:)*ybar;
  y_defl_x(xi,:) = cumsum(y_x(xi,:))*T;
end

%% Recover eigenvalues from spatial output
% extract time frame and combine velocity and deflection 
y = real([y_x(:,10:3000);y_defl_x(:,10:3000)]);

% consecutive time steps
y1 = y(:,1:end-1); 
y2 = y(:,2:end);

AA = y1 / y2; % recover state transition matrix
poles = eig(AA); % eigenvalues are poles
figure(); hold on; grid on;
az = diag(Az);
scatter(real(az),imag(az),'o')
scatter(real(poles),imag(poles),'x')
xlabel('Real part')
ylabel('Imaginary part')
legend({'True poles','Recovered poles'})


function x = clip(x,r)

x(x>r(2)) = r(2);
x(x<r(1)) = r(1);

end