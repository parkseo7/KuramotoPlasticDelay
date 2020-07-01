function sol = solvemodel2D(par, ddeopts)

    % Parameters
    w0 = par.w0 ;
    omega = w0*[1;1] ;
    g = par.g/2;
    
    kappa = par.gain ;
    tau0 = par.tau0 ; % [0.1;0.1]
    alphar = par.alphatau ;
    epsilon = par.epsilon ;
    
    t0 = par.t0 ;
    tf = par.tf ;
    
    histX = par.hist;
    
    % initial condition
    hist_lin = @(t) packX(histX(t-t0).', [tau0, tau0]) ;
    
    % Functions
    kuraf = @(t,X,Z) modelrhs(t,X,Z,omega,g,kappa,alphar,tau0,epsilon) ;
    tauf = @delays ;
    
    % solve
    sol = ddesd(kuraf, tauf, hist_lin, [t0,tf], ddeopts) ;
    sol.tau0 = tau0 ;
    sol.phi0 = histX(0);
   
end

function X = packX( theta, tau )
    X = [ theta(:) ; tau(:) ];
end

function dXdt = modelrhs(t,X,Z,omega,g,kappa,alphar,tau0,epsilon)
    theta = X(1:2);
    tau = X(3:end) ;
    Delta = [theta(2) - theta(1); theta(1) - theta(2)]; 
    thetadelay = [Z(2,1); Z(1,2)];
    dthetadt = omega + g*sin(thetadelay - theta);
    % dtaudt = alphar*posind(tau).*( -(tau - tau0) + kappa*Delta);
    dtaudt = alphar*posind(tau,epsilon).*( -(tau - tau0) + kappa*sin(Delta));
    dXdt = packX( dthetadt, dtaudt );
end

function d = delays(t,X)
    tau = X(3:end);
    d = t - tau(:);
end

function u = posind(tau, epsilon)
% Returns a vector with each component being 1 if tau_j > 0 and 0
% otherwise

u = double(tau > 0);
tau_inds = find((tau > 0).*(tau < epsilon));

% For each index in x_inds, replace x_pos:
MOL = @(t) exp(-(t-1).^-2).*exp(-(t+1).^-2);
MOL2 = @(v) MOL(-1 + 2*v/epsilon);
INT = integral(@(s) MOL2(s), 0, epsilon);

for i = 1:numel(tau_inds)
    u(tau_inds(i)) = integral(@(s) MOL2(s), 0, tau(tau_inds(i))) / INT;
end
end
