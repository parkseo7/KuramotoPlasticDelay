% Export directory
foldername = 'matlab_fig7_np' ;

% Set up directory (check if it exists)
cwd = pwd ;
dir_folder = fullfile(cwd, 'data', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Parameters
par = struct ;

par.N = 30 ; % 50 ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.tau0 = 2.0 ;
par.gain = 0 ; % 80 ;
par.alphatau = 1.0 ;
par.t_inj = 150;
par.t0 = 0 ;
par.tf = 300 ;

% History function
N = par.N;
std = 0.5;
init_freq = par.w0;
init_freqs = init_freq*ones(1,N);

T = sqrt(3)*std;
phases = T*(0:N-1)/N - T/2; % phases = T*rand(1,N) - T/2;

par.hist = IVPhistory(init_freqs, phases, par);

% Injury
inj_list = (0:9)/10;
n_trials = size(inj_list,2);

% DDE options
ddeopts = ddeset() ;
ddeopts.OutputFcn = @ddewbar;
% ddeopts.MaxStep = 1.0 ;

% Wait bar
f = waitbar(0,'Starting trials...') ;

% MAIN LOOP
for k = 1:n_trials
    inj = inj_list(k);
    par.inj = inj;
    
    % Waitbar
    waittext = ['Injury = ' num2str(inj)] ;
    waitprog = k / n_trials ;
    waitbar(waitprog, f, waittext) ;
    
    % Solve model
    sol = solvemodelinj(par, ddeopts) ;

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
    A = sol.A_inj;
    t_inj = par.t_inj;

    % Save file
    filename = ['inj_' num2str(k) '.mat'] ;
    dir_file = fullfile(dir_folder, filename) ;
    save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'tau0', 'N', 'gain', 'omega0', ...
        'T', 'g', 'tf', 'phi0', 'std', 'init_freq', 'A', 'inj', 't_inj')
end

close(f)
