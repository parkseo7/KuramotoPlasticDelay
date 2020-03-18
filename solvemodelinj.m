function sol = solvemodelinj(par, ddeopts)

    % Parameters
    N = par.N ;
    w0 = par.w0 ;
    omega = w0*ones(N,1) ;
    g = par.g ;
    tau0_0 = par.tau0 ;
    kappa = par.gain ;
    alphar = par.alphatau ;
    inj = par.inj ;
    t_inj = par.t_inj;
    t0 = par.t0 ;
    tf = par.tf ;
    
    % frequencies, connections, baseline conduction delays
    connprob = 1 - inj;
    a = double(rand(N,N)<connprob);
    tau0 = tau0_0*ones(N,N);
    % tau0 = 2*T*rand(N,N);
    A_inj = g/N*a;
    A = g/N*ones(N,N);
    
    % initial condition (constantly distributed around half-circle at t0)
    histX = par.hist;
    % hist_linX = @(t) offset*(pi/N)*(0:N-1) + omega.'*(t - t0) ;
    % hist_linX = @(t) (pi/N)*(0:N-1) + omega.'*(t - t0) ;
    hist_lin = @(t) packX(histX(t-t0), tau0) ;
    
    % Functions
    kuraf = @(t,X,Z) modelrhs(t,X,Z,omega,A,A_inj,kappa,alphar,tau0,t_inj) ;
    tauf = @delays ;
    
    % solve
    sol = ddesd(kuraf, tauf, hist_lin, [t0,tf], ddeopts) ;
    sol.tau0 = tau0 ;
    sol.A_inj = a;
   
end

function X = packX( theta, tau )
    X = [ theta(:) ; tau(:) ];
end

function [theta, tau, N] = unpackX( X )
    N = round( 0.5*(sqrt(4*numel(X)+1)-1) );
    theta = X(1:N);
    tau = reshape( X(N+1:end), [N,N] );
end

function dXdt = modelrhs(t,X,Z,omega,A,A_inj,kappa,alphar,tau0,t_inj)
    [theta, tau, N] = unpackX( X );
    thetadelay = Z(1:N,:);
    thetadelay = reshape(thetadelay(kron(eye(N),ones(1,N))==1),N,N);
    
    % Change to injury topology A_inj at t = t_inj
    if t < t_inj
        dthetadt = omega + sum( A.*sin( thetadelay - repmat(theta,1,N)), 2);
    else
        dthetadt = omega + sum( A_inj.*sin( thetadelay - repmat(theta,1,N)), 2);
    end
    
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