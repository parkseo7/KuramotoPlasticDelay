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
    epsilon = par.epsilon;
    
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
    kuraf = @(t,X,Z) modelrhs(t,X,Z,omega,A,A_inj,kappa,alphar,tau0,epsilon,t_inj) ;
    tauf = @delays ;
    
    % solve
    sol = ddesd(kuraf, tauf, hist_lin, [t0,tf], ddeopts) ;
    sol.tau0 = tau0 ;
    sol.A_inj = a;
    
    [~, tau_asy, ~] = unpackX(sol.y(:,end));
    sol.tau_top = a.*tau_asy;
    
end


function X = packX( theta, tau )
    X = [ theta(:) ; tau(:) ];
end

function [theta, tau, N] = unpackX( X )
    N = round( 0.5*(sqrt(4*numel(X)+1)-1) );
    theta = X(1:N);
    tau = reshape( X(N+1:end), [N,N] );
end

function dXdt = modelrhs(t,X,Z,omega,A,A_inj,kappa,alphar,tau0,epsilon,t_inj)
    [theta, tau, N] = unpackX( X );
    thetadelay = Z(1:N,:);
    thetadelay = reshape(thetadelay(kron(eye(N),ones(1,N))==1),N,N);
    
    % Change to injury topology A_inj at t = t_inj
    if t < t_inj
        dthetadt = omega + sum( A.*sin( thetadelay - repmat(theta,1,N)), 2);
    else
        dthetadt = omega + sum( A_inj.*sin( thetadelay - repmat(theta,1,N)), 2);
    end
    
    dtaudt = alphar*posind(tau,epsilon).*( -(tau - tau0) + kappa*sin(bsxfun(@minus,theta',theta)));
    dXdt = packX( dthetadt, dtaudt );
    
    disp(['Time = ' num2str(t) ', min = ' num2str(min(tau, [], 'all')), ', max = ' num2str(max(tau, [], 'all'))])
end

function d = delays(t, X)
    [~, tau] = unpackX( X );
    tau_pos = (tau > 0) .* tau; % Ensure positivity
    d = t - tau_pos(:);
end

function u = posind(tau, epsilon)
% Returns a vector with each component being 1 if tau_j > 0 and 0
% otherwise

u = double(tau >= epsilon);

% For each index in x_inds, replace x_pos:
MOL = @(t) exp(-(t-1).^-2).*exp(-(t+1).^-2);
MOL2 = @(v) MOL(-1 + 2*v/epsilon);
INT = integral(@(s) MOL2(s), 0, epsilon);

% With 0 < tau < epsilon, choose the average value.
tau_inds1 = find((tau > 0).*(tau < epsilon));

if numel(tau_inds1) > 0
    tau_avg = mean(tau(tau_inds1));
    u_avg = integral(@(s) MOL2(s), 0, tau_avg) / INT;
    
    u(tau_inds1) = u_avg ;
end
end
