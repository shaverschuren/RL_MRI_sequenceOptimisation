%% Reconsocket Matlab Interface to process images
% This script simulates the behaviour of the scanner for
% prototyping purposes

% Some setup parameters
clearvars;clc;
cd('/nfs/arch11/researchData/USER/tbruijne/Projects_Main/ReconSocket/recon-socket-repo/recon-socket/matlab_scripts/caspr/utils')
% run caspr_machine_setup.m
% par = check_default_values(par);
N = 64;
text_file_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/machine_flip_angles.txt';
data_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/img_data.h5';

% Create database (if applicable)
if ~exist(data_loc)
    h5create(data_loc,'/img',[N N])
end

% While loop which takes and proccesses an image
while 1    
    if exist(text_file_loc)
	% Generate random image
        img = rand(1)*ones(N,N) + 0.05*rand(N,N);
	% Write image to data drop
        h5write(data_loc,'/img',img);
	% Remove text file to signal completion
	delete(text_file_loc)

    else
        disp(['RMI: waiting...']);pause(0.1);
    end
           
end
