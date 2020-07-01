% Set up directory (check if it exists)
foldername = 'matlab_fig4_4' ;
cwd = pwd ;
dir_folder = fullfile(cwd, 'data2', foldername) ;

if ~exist(dir_folder, 'dir')
   mkdir(dir_folder)
end

% Constant parameters
par = struct ;
par.w0 = 1.0 ;
par.g = 1.5 ;
par.t0 = 0 ;
par.tf = 80 ;
par.gain = 30 ;
par.alphatau = 0.5 ;
par.tau0 = 0.1;
par.epsilon = 0.01;
par.A = [0 1; 1 0];

% Export names
g = par.g ;
gain = par.gain ;
omega0 = par.w0 ;
tau0 = par.tau0.' ;
t0 = par.t0 ;
tf = par.tf ;


% Varying parameters
n_freq = 13;
n_Delta = 10;
L_Delta = 1.0;
L_freq = 0.6;
Delta_arr = L_Delta * linspace(0, L_Delta, n_Delta+1);
Delta_arr = Delta_arr + Delta_arr(2);
Delta_arr = Delta_arr(1:end-1);
freq_arr = linspace(omega0 - L_freq, omega0 + L_freq, n_freq);
% freq_arr = linspace(omega0 - L_freq * 2 * g, omega0 + L_freq * 2 * g, n_freq);
% freq_arr = rand(1,n_trials) - 0.5;
% freq_arr = par.w0 + L_freq * 2 * g * freq_arr;

total = n_freq * n_Delta;

% DDE options
ddeopts = ddeset() ;
% ddeopts.MaxStep = 1.0 ;

% Wait bar
f = waitbar(0,'Starting trials...') ;

% MAIN LOOP
for j = 1:n_freq
    for k = 1:n_Delta

        Delta0 = Delta_arr(k);
        init_freq = freq_arr(j);
        num = (j-1) * n_Delta + k;
        
        % Waitbar
        waittext = ['Delta = ' num2str(Delta0) ', Init.freq = ' num2str(init_freq)] ;
        waitprog = num / (n_freq * n_Delta);
        waitbar(waitprog, f, waittext) ;

        % Solve model
        % par.hist = IVPhistory2(init_freq, Delta0, par);
        par.hist = @(t) t*[init_freq ; init_freq] + [0 ; Delta0];
        sol = solvemodel2D(par, ddeopts) ;

        % Export (transpose all matrices)
        t = sol.x.' ;
        y = sol.y(1:2,:).' ;
        yp = sol.yp(1:2,:).' ;

        tau = sol.y(3:end,:).' ;
        taup = sol.yp(3:end,:).' ;

        % Save file
        filename = ['2D_num_' num2str(num) '.mat'] ;
        dir_file = fullfile(dir_folder, filename) ;
        save(dir_file, 't', 'y', 'yp', 'tau', 'taup', 'tau0', 'gain', 'omega0', ...
            'g', 'tf', 'Delta0', 'init_freq')
    end
end

close(f)

