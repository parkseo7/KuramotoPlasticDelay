% Set up directory (check if it exists)
foldername = 'matlab_fig5' ;
cwd = pwd ;
dir_folder = fullfile(cwd, 'data', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Parameters
par = struct ;

par.N = 50 ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.alphatau = 1.0 ;
par.inj = 0.0 ;
par.t0 = 0 ;
par.tf = 50 ;
par.gain = 100;
par.tau0 = 0.1;
par.epsilon = 0.01;
par.A = ones(par.N);

N = par.N;
gain = par.gain;
omega0 = par.w0;
tau0 = par.tau0;
tf = par.tf ;
g = par.g ;

% Varying parameters
n_trials = 1; % Increase to 10
L_std = 0.5;
L_freq = 0.25; % Multiple of g
std_arr = L_std*rand(1,n_trials);
freq_arr = rand(1,n_trials) - 0.5;

freq_arr = par.w0 + L_freq * 2 * g * freq_arr;

% DDE options
ddeopts = ddeset() ;
ddeopts.NormControl = 'on';
ddeopts.OutputFcn = @ddewbar;
% ddeopts.MaxStep = 1.0 ;

% Wait bar
f = waitbar(0,'Starting trials...') ;

% MAIN LOOP
for k = 1:n_trials
    std = std_arr(k);
    freq = freq_arr(k);
        
    % Waitbar
    waittext = ['trial: ' num2str(k) ', std = ' num2str(std) ', freq = ' num2str(freq)] ;
    waitprog = k / n_trials ;
    waitbar(waitprog, f, waittext) ;

    % History function
    T = sqrt(3)*std;
    phases = T*rand(1,N) - T/2;
    init_freq = freq;
    init_freqs = init_freq*ones(1,N);

    par.hist = IVPhistory(init_freqs, phases, par);

    % Solve model
    sol = solvemodel(par, ddeopts) ;

    % Export (transpose all matrices)
    t = sol.x.' ;
    y = sol.y(1:N,:).' ;
    yp = sol.yp(1:N,:).' ;

    tau = sol.y(N+1:end,:).' ;
    taup = sol.yp(N+1:end,:).' ;

    phi0 = phases;

    % Save file
    filename = ['sol_num' num2str(k) '.mat'] ;
    dir_file = fullfile(dir_folder, filename) ;
    save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'N', 'gain', 'omega0', ...
        'tau0', 'phi0', 'g', 'tf', 'std', 'init_freq')
end

close(f)
