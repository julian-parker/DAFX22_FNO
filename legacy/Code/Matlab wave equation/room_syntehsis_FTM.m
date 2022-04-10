%% Physical Model of a 2D room 
%
%   Synthesis is based on the Functional Transformation Method (FTM)
% 
%   Dr. Maximilian Schaefer @FAU, 2021
%   Contact: max.schaefer@fau.de
%
%   References: ﻿
% Schäfer, M., Rabenstein, R., & Schlecht, S. J. (2020). A String in a Room: 
%   Mixed-Dimensional Transfer Function Models for Sound Synthesis. 
%   23rd International Conference on Digital Audio Effects (DAFx2020), 
%   September, 24–30.


%% Basic simulation param 
% Sampling frequency 
Fs = 48000; 
T = 1/Fs;
% Simulation duration 
dur = 0.1;
t = 0:T:dur-T;

%% Room Parameters 
lx = 2;
ly = 1;

c0 = 340;
rho = 1.2041;

% artificial damping factor. Maybe interesting to change during learning or
% estimation? 
damping = 1; 

%% FTM - Parameters
Mux = 10; 
Muy = 10; 

% Create index vector. 
% We have modes in 2 directions x,y but we want to count the eigenvalues
% with one index 'mu' --> each 'mu' is a index tupel 
index = zeros(Mux*Muy,2);
index(:,2) = repmat(1:Muy,1,Mux);
index(:,1) = repelem(1:Mux, Muy);

%[X,Y] = ndgrid(1:Mux,1:Muy);
%[X(:),Y(:)]

Mu = Mux*Muy;
mu = 1:Mu;
mux = index(mu,1);
muy = index(mu,2);
smu = zeros(1,Mu);

%% FTM - Eigenvalues 

% create wavenumbers 
lamX = mux.*pi./lx;
lamY = muy.*pi./ly;

% create eigenvalues 
smu(mu) = 1j*c0*sqrt(lamX(mu).^2 + lamY(mu).^2);

% Now condsider complex conjugated eigenvalues 
smu = [smu (conj(smu))];
lamX = [lamX.' lamX.'];
lamY = [lamY.' lamY.'];
Mu = length(smu);

% Add the artificial damping 
smu = smu - damping; 



%% FTM - scaling factor 
nx = -8.*lamX.^2./(rho*smu.^2)*lx*ly;
ny = -8.*lamY.^2./(rho*smu.^2)*lx*ly;
nc = 8/(rho*c0^2)*lx*ly;

nmu = nx+ny+nc;

%% FTM - Eigenfunctions 
K1 = @(x,y) 4*cos(lamX.*x).*cos(lamY.*y);
K2 = @(x,y) 4*lamX./(smu*rho).*sin(lamX.*x).*cos(lamY.*y);
K3 = @(x,y) 4*lamY./(smu*rho).*cos(lamX.*x).*sin(lamY.*y);

Ka1 = @(x,y) -4*lamX./(smu*rho).*sin(lamX.*x).*cos(lamY.*y);
Ka2 = @(x,y) -4*lamY./(smu*rho).*cos(lamX.*x).*sin(lamY.*y);
Ka3 = @(x,y) 4*cos(lamX.*x).*cos(lamY.*y);

%% Input signal 

% Use a delta at exciation position on the string 
xe = 0.4*lx;
ye = 0.5*ly;

fe_xy = zeros(1,length(smu));

% impulse excitation at (xe,ye)
fe_x = Ka3(xe,ye);

% % Add random excitation 
% for mu = 1:Mu
%     x = 0:1e-3:lx; 
%     y = 0:1e-3:ly; 
%     
%     Kx = 4*cos(lamX(mu).*x); 
%     Ky = cos(lamY(mu).*y);
% 
%     funX = 4*cos(lamX(mu).*x).*(rand(1,length(x))*2-1);
%     funY = cos(lamY(mu).*y).*(rand(1,length(y))*2-1);
% 
%     fe_x(mu) = trapz(x,funX)*trapz(y,funY);
% end



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

%% Simulation - output equation 

xo = 0.7*lx; 
yo = 0.7*ly; 

% take first entry (sound pressure) 
y = (K1(xo,yo)./nmu)*ybar;

% hear it 
y = real(y); 
% y = y/max(y); 
% sound(y,Fs); 

%% Simulation - spatial domain 

% create spatial grid; 
% delta = 1e-3; 
% xs = linspace(0,lx,50); 
% ys = linspace(0,ly,50); 
% 
% K1_sp = zeros(length(xs), length(ys), Mu);
% y_sp = zeros(length(xs), length(ys), length(t));
% 
% for xi = 1:length(xs) 
%     for yi = 1:length(ys) 
%         K1_sp(xi,yi,:) = K1(xs(xi), ys(yi))./nmu;
%         y_sp(xi,yi,:) = squeeze(K1_sp(xi,yi,:)).'*ybar;
%     end
% end



%% Simulation - spatial domain 
% create spatial grid; 
delta = 1e-3; 
xs = linspace(0,lx,50); 
ys = linspace(0,ly,50); 
K1_sp = zeros(length(xs), length(ys), Mu);  % Eigenfunctions for sound pressure 
K2_vx = zeros(length(xs), length(ys), Mu);  % Eigenfunctions for particle velocity in x-direction 
K3_vy = zeros(length(xs), length(ys), Mu);  % Eigenfunctions for particle velocity in y-direction 

y_sp = zeros(length(xs), length(ys), length(t));    % Output sound pressure
y_vx = zeros(length(xs), length(ys), length(t));    % output particle velocity in x-direction 
y_vy = zeros(length(xs), length(ys), length(t));    % output particle velocity in y-direction 
for xi = 1:length(xs) 
    for yi = 1:length(ys) 
        K1_sp(xi,yi,:) = K1(xs(xi), ys(yi))./nmu;
        K2_vx(xi,yi,:) = K2(xs(xi), ys(yi))./nmu;
        K3_vy(xi,yi,:) = K3(xs(xi), ys(yi))./nmu;
        
        y_sp(xi,yi,:) = squeeze(K1_sp(xi,yi,:)).'*ybar;
        y_vx(xi,yi,:) = squeeze(K2_vx(xi,yi,:)).'*ybar;
        y_vy(xi,yi,:) = squeeze(K3_vy(xi,yi,:)).'*ybar;
    end
end


%% visualize spatial domain through time
figure()
img = real(y_sp);
plt = imagesc(img(:,:,1));
for t = 1:size(img,3)
    plt.set('CData',img(:,:,t));
    drawnow
end

