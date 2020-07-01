function phi = IVPhistory2(freq, Delta, par)
% Returns a history function (with respect to t < 0, with the following
% regions:
% -tau0 < t <= 0: hermite interpolated polynomial
% t < -tau0: linear function with slope freqs
% t = 0: phi(0) = phases
% Here, freqs and phases are 1D row arrays of the same length.

% Parameters
tau0 = par.tau0;
g = par.g;
w0 = par.w0;

% Determine phase locations at t = -tau0:
phases0 = asin((freq - w0) / g) * ones(1,2) + [Delta 0];

% Hermite interpolation
phi = @(t) cubic_int(t, -tau0, phases0.', 0, [0 ; Delta], [0 ; 0], [freq ; freq]).';

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

