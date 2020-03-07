% Parameters
par = struct();
par.g = 1.5;
par.tau0 = 0.1;
par.omega0 = 1.0;

N = 10;
phases = (1:N)/N;
freqs = 1.5*ones(1,N);

phi = IVPhistory(freqs, phases, par);

t = linspace(-0.3,0,100);
y = zeros(N,size(t,2));

X = phi(-0.5);

for i = 1:100
    tnew = t(i);
    y(:,i) = phi(tnew);
end

% Plot
figure
plot(t, y)
