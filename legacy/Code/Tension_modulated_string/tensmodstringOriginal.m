%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Tension Modulated String
%
% [1] Trautmann, Rabenstein, Sound Synthesis with Tension Modulated
% Nonlinearities Based on Functional Transformations
% AMTA 2000, Jamaica
%
% R. Rabenstein, 27.03.2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameter Values from [1] arranged as structure P

P.ell = 1;            % m             string length at rest
P.A   = 0.19634e-6;   % m^2           string cross section area
P.I   = 0.02454e-12;  % m^4           string moment of intertia
P.rho = 7800;         % kg/m^3        string density
P.E   = 190e9;        % Pa            string elasticity
P.d1  = 4e-3;         % kg/(ms)       string frequ. independent loss
P.d3  = 6e-5;         % kg m/s        string frequ. dependent loss
P.Ts0 = 150;          % N             string tension
P.M   = 50;           %               number of expansion terms

xe  = 0.28;         % m             pluck position
hi  = 0.03;         % m             initial deflection at pluck position
xa  = 0.28;         % m             listening position

tmax = 0.1;         % s             time duration for evaluation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tspan   = [0 tmax];                    % time span for ode

mu      = [1:P.M]';                     % index for Fourier-Sine transf.
kmu     = mu*pi/P.ell;                  % argument of sine functions

% Fourier-Sine coefficients of the initial deflection
yb0     = hi *(P.ell/(P.ell-xe)*sin(kmu*xe)./(kmu*xe))./kmu;

zb0     = zeros(P.M,1);               % initial condition for derivative 
wb0     = [yb0; zb0];                 % initial condition for ode   

[t1,wb1] = ode45(@(t1,wb1) tensmodstr(t1,wb1,P),tspan,wb0); % solve ode

yb1 = wb1(:,[1:P.M]);   % solution for the FS-transform of the deflection
                   
% inverse Fourier-Sine transformation
y1      = yb1*sin(kmu*xa) * 2/P.ell;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% comparison with initial deflection of 10% of the previous value

yb0      = yb0/10;
zb0      = zeros(P.M,1);               % initial condition for derivative 
wb0      = [yb0; zb0];                 % initial condition for ode   
[t2,wb2] = ode45(@(t2,wb2) tensmodstr(t2,wb2,P),tspan,wb0); % solve ode
yb2      = wb2(:,[1:P.M]);   % solution for the FS-transform of the deflection
y2       = yb2*sin(kmu*xa) * 2/P.ell; % inverse Fourier-Sine transformation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot(t1,y1,'k-',t2,10*y2,'c-')  % magnify yb for comparison
title('Tension Modulated String'); grid
xlabel('Time t in seconds');
ylabel('deflection y in m');
legend('strong pluck','weak pluck','location','southwest')
% strong pluck: initial value with deflection hi at pluck position xe
% weak pluck:   initial value is 10% of the strong pluck and magnified
%               by a factor of 10 in the plot for comparison
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dwb_dt = tensmodstr(t,wb,P)

% extract parameter values from structure P
ell     = P.ell;
E       = P.E;
I       = P.I;
rho     = P.rho;
A       = P.A;
d1      = P.d1;
d3      = P.d3;
Ts0     = P.Ts0;
M       = P.M;

% extract yb and zb from wb
yb      = wb(1:M);
zb      = wb(M+1:2*M);

% set up vectors and matrices
mu      = [1:M]';
kmu     = mu*pi/ell;
Mz      = diag((d1+d3*kmu.^2)/(rho*A));
My      = diag((Ts0*kmu.^2 + E*I*kmu.^4)/(rho*A));
M1      = diag(kmu.^2/(rho*A));
M2      = diag(mu.^2);

% calculate additional string tension Ts1
Ts1     = E*A*pi^2/ell^4 * yb'*M2*yb; 

% calculate first order derivatives
 
dyb_dt  = zb;
dzb_dt  = - Mz*zb - My*yb;      % linear terms
dzb_dt  = dzb_dt - M1*Ts1*yb;   % nonlinear term
dwb_dt  = [dyb_dt; dzb_dt];
end