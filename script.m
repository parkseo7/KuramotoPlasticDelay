% Export directory
foldername = 'matlab1' ;
trial = 1;

% Parameters
par = struct ;

par.N = 50 ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.T = 0.1 ;
par.gain = 30 ;
par.alphatau = 1.0 ;
par.inj = 0.0 ;
par.t0 = 0 ;
par.tf = 30 ;
   
% DDE options
ddeopts = ddeset() ;
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
tau0 = sol.tau0 ;
phi0 = sol.phi0 ;

gain = par.gain ;
omega0 = par.w0 ;
tf = par.tf ;
g = par.g ;
T = par.T ;

% Set up directory (check if it exists)
cwd = pwd ;
dir_folder = fullfile(cwd, 'data', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Save file
filename = ['sol' num2str(N) '_gain' num2str(gain) '_' num2str(trial) '.mat'] ;
dir_file = fullfile(dir_folder, filename) ;
save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'tau0', 'N', 'gain', 'omega0', ...
    'T', 'g', 'tf', 'phi0')
