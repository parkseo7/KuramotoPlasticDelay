% Set up directory (check if it exists)
foldername = 'matlab2' ;
cwd = pwd ;
dir_folder = fullfile(cwd, 'data', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Array of gain, T:
gain_arr = 0:5:5 ;
T_arr = 0:0.1:0.1 ;

total = numel(gain_arr)*numel(T_arr) ;

% Parameters
par = struct ;

par.N = 30 ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.alphatau = 1.0 ;
par.inj = 0.0 ;
par.t0 = 0 ;
par.tf = 30 ;

% DDE options
ddeopts = ddeset() ;
% ddeopts.MaxStep = 1.0 ;

% Wait bar
f = waitbar(0,'Starting trials...') ;

% MAIN LOOP
waitk = 0 ;
for gain = gain_arr
    for T = T_arr
        
        % Waitbar
        waittext = ['gain = ' num2str(gain) ', T = ' num2str(T)] ;
        waitprog = waitk / total ;
        waitbar(waitprog, f, waittext) ;
       
        % Variable parameters
        par.T = T ;
        par.gain = gain ;

        % Solve model
        sol = solvemodel(par, ddeopts) ;

        % Export (transpose all matrices)
        N = par.N;
        t = sol.x.' ;
        y = sol.y(1:N,:).' ;
        yp = sol.yp(1:N,:).' ;

        tau = sol.y(N+1:end,:).' ;
        taup = sol.yp(N+1:end,:).' ;

        omega0 = par.w0 ;
        tf = par.tf ;
        g = par.g ;

        % Save file
        filename = ['sol_T' erase(num2str(T),'.') '_gain' num2str(gain) '.mat'] ;
        dir_file = fullfile(dir_folder, filename) ;
        save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'N', 'gain', 'omega0', ...
            'T', 'g', 'tf')
        
        % Wait progress
        waitk = waitk + 1 ;
    end
end

close(f)
