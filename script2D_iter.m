% Set up directory (check if it exists)
foldername = 'matlab2D_multi5' ;
cwd = pwd ;
dir_folder = fullfile(cwd, 'data', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Constant parameters
par = struct ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.t0 = 0 ;
par.tf = 600 ;
par.gain = 30 ;
par.alphatau = 1.0 ;
par.tau0 = 0.1;

% Export names
g = par.g ;
gain = par.gain ;
omega0 = par.w0 ;
tau0 = par.tau0.' ;
t0 = par.t0 ;
tf = par.tf ;

% Varying parameters
Delta_arr = (0.1:0.1:1)*pi/2 ;
freq_arr = linspace(omega0-g/4,omega0+g/4,5);

total = numel(Delta_arr)*numel(freq_arr) ;

% DDE options
ddeopts = ddeset() ;
% ddeopts.MaxStep = 1.0 ;

% Wait bar
f = waitbar(0,'Starting trials...') ;

% MAIN LOOP
waitk = 0 ;
for Delta0 = Delta_arr
    for init_freq = freq_arr
        
        % Waitbar
        waittext = ['Delta = ' num2str(Delta0) ', Init.freq = ' num2str(init_freq)] ;
        waitprog = waitk / total ;
        waitbar(waitprog, f, waittext) ;
        
        % Solve model
        par.hist = IVPhistory([init_freq, init_freq], [0 Delta0], par);

        sol = solvemodel2D(par, ddeopts) ;

        % Export (transpose all matrices)
        t = sol.x.' ;
        y = sol.y(1:2,:).' ;
        yp = sol.yp(1:2,:).' ;

        tau = sol.y(3:end,:).' ;
        taup = sol.yp(3:end,:).' ;

        % Save file
        filename = ['2D_num_' num2str(waitk) '.mat'] ;
        dir_file = fullfile(dir_folder, filename) ;
        save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'tau0', 'gain', 'omega0', ...
            'g', 'tf', 'Delta0', 'init_freq')
        
        % Wait progress
        waitk = waitk + 1 ;
    end
end

close(f)

