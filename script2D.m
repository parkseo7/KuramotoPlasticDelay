foldername = 'matlab_fig4' ;
num = 1;

% Parameters
par = struct ;

par.w0 = 1.0 ;
par.g = 1.5 ;
par.init_freq = 1.0;

par.t0 = 0 ;
par.tf = 100 ;

par.gain = 80 ;
par.alphatau = 1.0 ;
par.tau0 = 0.1;

% History function
init_freq = 1.0;
Delta0 = pi/4;
par.hist = IVPhistory([init_freq, init_freq], [0 Delta0], par);

% DDE options
ddeopts = ddeset() ;
% ddeopts.MaxStep = 1.0 ;

% Solve model
sol = solvemodel2D(par, ddeopts) ;

% Export (transpose all matrices)
t = sol.x.' ;
y = sol.y(1:2,:).' ;
yp = sol.yp(1:2,:).' ;

tau = sol.y(3:end,:).' ;
taup = sol.yp(3:end,:).' ;

% Parameters (export)
g = par.g ;
gain = par.gain ;
omega0 = par.w0 ;

tau0 = par.tau0.' ;

t0 = par.t0 ;
tf = par.tf ;

% Set up directory (check if it exists)
cwd = pwd ;
dir_folder = fullfile(cwd, 'data2', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Save file
filename = ['gain_' num2str(gain) '_num_' num2str(num) '.mat'] ;
dir_file = fullfile(dir_folder, filename) ;
save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'tau0', 'gain', 'omega0', ...
    'g', 'tf', 'init_freq', 'Delta0')
