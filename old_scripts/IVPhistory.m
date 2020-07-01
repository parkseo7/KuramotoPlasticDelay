function phi = IVPhistory(freqs, phases, par)
% Returns a history function (with respect to t < 0, with the following
% regions:
% -tau0 < t <= 0: hermite interpolated polynomial
% t < -tau0: linear function with slope freqs
% t = 0: phi(0) = phases
% Here, freqs and phases are 1D row arrays of the same length.

% Parameters
[N,~] = size(phases);
tau0 = par.tau0;
g = par.g;
w0 = par.w0;
A = par.A;

% Define linear functions for t < -tau0
phi0 = @(t) t*freqs + phases;

% Use Kuramoto DDE to obtain derivatives at t = 0:
phi_tau0 = phases - freqs*tau0;
dphidt = w0 + (g/N) *  sum(A .* sin(phi_tau0 - phases.'),2).';

% Hermite interpolation
phi_int = @(t) cubic_int(t, -tau0, phi_tau0.', 0, phases.', freqs.', dphidt.').';
phi = @(t) hist_pw(t, -tau0, phi0, phi_int).';

end

function y = hist_pw(t, t0, fun1, fun2)

if t <= t0
    y = fun1(t);
else
    y = fun2(t);
end

end


function yint = cubic_int(tint,t,y,tnew,ynew,yp,ypnew)

h = tnew - t;
s = (tint - t)/h;
s2 = s .* s;
s3 = s .* s2;
slope = (ynew - y)/h;
c = 3*slope - 2*yp - ypnew;
d = yp + ypnew - 2*slope;
yint = y(:,ones(size(tint))) + (h*d*s3 + h*c*s2 + h*yp*s);        

end    

