%% Run the EPG simulations
addpath(genpath('../epg_code/matlab/'))

%% Very simple example
% Simulation/spin properties
T1      = 0.600;    % T1 relaxation time of the spin [s]
T2      = 0.300;    % T2 relaxation time of the spin [s]
B1      = 1;        % B1 field of the spin [-] 
fa      = 20;       % Flip angle of the sequence [deg] (Can also be an array)
Nfa     = 1000;      % Number of flip angles to achieve a steady state [-]
tr      = 5E-03;    % Repetition time [s]
spoiled = 1;        % 0 = balanced, 1 = spoiled

% Simulate a single spin
signal = EPG(Nfa,fa,1,tr,T1,T2,spoiled);

% Some basic visualization
figure;subplot(221);plot(abs(signal));title('|S|')
subplot(222);plot(real(signal));title('R(S)')
subplot(223);plot(imag(signal));title('I(S)')
subplot(224);plot(angle(signal));title('\angle S')

%% A bit more involved example with a random flip angle train
% Simulation/spin properties
T1      = .583;             % T1 relaxation time of the spin [s]
T2      = 0.055;            % T2 relaxation time of the spin [s]
B1      = 1;                % B1 field of the spin [-] 
fa      = 70*rand(500,1);   % Flip angle of the sequence [deg] (Can also be an array)
Nfa     = 1000;              % Number of flip angles to achieve a steady state [-]
tr      = 10E-03;           % Repetition time [s]
spoiled = 1;                % 0 = balanced, 1 = spoiled

% Simulate a single spin
signal = EPG(Nfa,fa,1,tr,T1,T2,spoiled);

% Some basic visualization
figure;subplot(221);plot(abs(signal));title('|S|')
subplot(222);plot(real(signal));title('R(S)')
subplot(223);plot(imag(signal));title('I(S)')
subplot(224);plot(angle(signal));title('\angle S')
