function sol = solvemodel(par, ddeopts)

    % Parameters
    N = par.N ;
    w0 = par.w0 ;
    omega = w0*ones(N,1) ;
    g = par.g ;
    T = par.T ;
    kappa = par.gain ;
    alphar = par.alphatau ;
    inj = par.inj ;
    t0 = par.t0 ;
    tf = par.tf ;

    % frequencies, connections, baseline conduction delays
    connprob = 1 - inj;
    a = double(rand(N,N)<connprob);
    tau0 = 2*T*rand(N,N);
    A = g/N*a;
    
    % initial condition (constantly distributed around half-circle at t0)
    hist_linX = @(t) (pi/N)*(0:N-1) + omega.'*(t - t0) ;
    hist_lin = @(t) packX(hist_linX(t), tau0) ;
    
    % Functions
    kuraf = @(t,X,Z) modelrhs(t,X,Z,omega,A,kappa,alphar,tau0) ;
    tauf = @delays ;
    
    % solve
    sol = ddesd(kuraf, tauf, hist_lin, [t0,tf], ddeopts) ;
   
end

function X = packX( theta, tau )
    X = [ theta(:) ; tau(:) ];
end

function [theta, tau, N] = unpackX( X )
    N = round( 0.5*(sqrt(4*numel(X)+1)-1) );
    theta = X(1:N);
    tau = reshape( X(N+1:end), [N,N] );
end

function dXdt = modelrhs(t,X,Z,omega,A,kappa,alphar,tau0)
    [theta, tau, N] = unpackX( X );
    thetadelay = Z(1:N,:);
    thetadelay = reshape(thetadelay(kron(eye(N),ones(1,N))==1),N,N);
    dthetadt = omega + sum( A.*sin( thetadelay - repmat(theta,1,N)), 2);
    dtaudt = alphar*posind(tau).*( -(tau - tau0) + kappa*bsxfun(@minus,theta',theta));
    dXdt = packX( dthetadt, dtaudt );
end

function d = delays(t,X)
    [~, tau] = unpackX( X );
    d = t - tau(:);
end

function u = posind(tau)
% Returns a vector with each component being 1 if tau_j > 0 and 0
% otherwise

u = (tau > 0).*tau ;
end