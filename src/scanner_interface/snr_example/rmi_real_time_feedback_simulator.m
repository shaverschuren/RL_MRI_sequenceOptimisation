%% Reconsocket Matlab Interface to process images
% This script downloads the images on the RMQ exchange server using our mex
% interface.

% Some setup parameters
clearvars;clc;
cd('/nfs/arch11/researchData/USER/tbruijne/Projects_Main/ReconSocket/recon-socket-repo/recon-socket/matlab_scripts/caspr/utils')
run caspr_machine_setup.m
par = check_default_values(par);
N = 64;
text_file_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/machine_flip_angles.txt';
global signal_prev;
global fa_prev;
global dfa;
global iter;

% While loop which takes and proccesses an image
%rmi_maximize_snr(); % Initialize database setup
signal_prev = 0;
fa_prev = 40;
dfa = -1;
iter = 0;
img_store = [];
while 1    
    if ~exist(text_file_loc)
        img = rand(1)*ones(N,N) + 0.05*rand(N,N);
        rmi_maximize_snr(img,text_file_loc);
    else
        disp(['RMI: waiting...']);pause(1);
    end
           
end