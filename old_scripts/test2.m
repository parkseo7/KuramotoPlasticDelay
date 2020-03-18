% Parameters
par = struct ;

par.N = 30 ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.tau0 = 0.1 ;
par.gain = 30 ;
par.alphatau = 1.0 ;
par.inj = 0.0 ;
par.t0 = 0 ;
par.tf = 300 ;

% History function
N = par.N;
std = 0.25;
init_freq = 0.8;
init_freqs = init_freq*ones(1,N);

T = sqrt(3)*std;
phases = T*rand(1,N) - T/2;

par.hist = IVPhistory(init_freqs, phases, par);

phi = par.hist;

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
