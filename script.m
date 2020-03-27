% Export directory
foldername = 'matlab_ND' ;
trial = 1;

% Parameters
par = struct ;

par.N = 50 ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.tau0 = 0.1 ;
par.gain = 80 ;
par.alphatau = 1.0 ;
par.inj = 0.0 ;
par.t0 = 0 ;
par.tf = 90 ;

% History function
N = par.N;
std = 0.75;
init_freq = par.w0;
init_freqs = init_freq*ones(1,N);

T = sqrt(3)*std;
phases = T*rand(1,N) - T/2;

par.hist = IVPhistory(init_freqs, phases, par);

% DDE options
ddeopts = ddeset() ;
ddeopts.OutputFcn = @ddewbar;
% ddeopts.MaxStep = 1.0 ;

% Solve model
sol = solvemodel(par, ddeopts) ;

% Export (transpose all matrices)
N = par.N;
t = sol.x.' ;
y = sol.y(1:N,:).' ;
yp = sol.yp(1:N,:).' ;

tau = sol.y(N+1:end,:).' ;
taup = sol.yp(N+1:end,:).' ;
phi0 = phases;

gain = par.gain ;
omega0 = par.w0 ;
tf = par.tf ;
g = par.g ;
tau0 = par.tau0;

% Set up directory (check if it exists)
cwd = pwd ;
dir_folder = fullfile(cwd, 'data', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Save file
filename = ['sol_' num2str(trial) '.mat'] ;
dir_file = fullfile(dir_folder, filename) ;
save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'tau0', 'N', 'gain', 'omega0', ...
    'T', 'g', 'tf', 'phi0', 'std', 'init_freq')
