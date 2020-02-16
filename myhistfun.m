function hist = myhistfun(N, t0, par, mode)
% Generates a custom history function for our numerical purposes.

if strcmp(mode, 'linear')
    hist = linear(N, t0, par) ;
else
    hist = @(t) zeros(1,N) ;
end


end

function hist = linear(N, t0, par)
    freq = par.freq ;
    p0 = par.phase0 ;
    pf = par.phasef ;
    if strcmp(par.phases, 'uniform')
        phases = p0 + (pf - p0)*(0:N-1)/N ;
    elseif strcmp(par.phases, 'random')
        phases = p0 + (pf - p0)*rand(1,N) ;
    else
        phases = zeros(1,3) ;
    end
    
    hist = @(t) phases + freq.'*(t - t0) ;
    
end


