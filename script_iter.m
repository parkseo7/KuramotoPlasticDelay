% Set up directory (check if it exists)
foldername = 'matlabND_multi_N30_4' ;
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
par.tf = 80 ;
par.gain = 60;
par.tau0 = 0.1;

N = par.N;
gain = par.gain;
omega0 = par.w0;
tau0 = par.tau0;
tf = par.tf ;
g = par.g ;
        
% Varying parameters
L_std = 1.0;
n_std = 4;
L_freq = 1.0;
n_freq = 4;
std_arr = L_std*rand(1,n_std);% (0.1:0.2:1)*pi/4 ;
freq_arr = L_freq*(rand(1,n_freq) - 0.5); % linspace(omega0-L_freq,omega0+L_freq,2);
freq_arr = par.w0 + freq_arr;

total = numel(std_arr)*numel(freq_arr) ;

% DDE options
ddeopts = ddeset() ;
% ddeopts.MaxStep = 1.0 ;

% Wait bar
f = waitbar(0,'Starting trials...') ;

% MAIN LOOP
waitk = 0 ;
for std = std_arr
    for freq = freq_arr
        
        % Waitbar
        waittext = ['std = ' num2str(std) ', freq = ' num2str(freq)] ;
        waitprog = waitk / total ;
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
        filename = ['sol_num' num2str(waitk) '.mat'] ;
        dir_file = fullfile(dir_folder, filename) ;
        save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'N', 'gain', 'omega0', ...
            'tau0', 'phi0', 'g', 'tf', 'std', 'init_freq')
        
        % Wait progress
        waitk = waitk + 1 ;
    end
end

close(f)
