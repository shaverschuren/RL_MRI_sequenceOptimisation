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
data_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/img_data.h5';

% Create database (if applicable)
if ~exist(data_loc)
    h5create(data_loc,'/img',[N N])
end

% While loop which takes and proccesses an image
while 1    
    if ~exist(text_file_loc)
        img = rand(1)*ones(N,N) + 0.05*rand(N,N);
        h5write(data_loc,'/img',img);
    else
        disp(['RMI: waiting...']);pause(1);
    end
           
end