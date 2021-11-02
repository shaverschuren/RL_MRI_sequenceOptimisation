%% Reconsocket Matlab Interface to process images
% This script simulates the behaviour of the scanner for
% prototyping purposes.

% Some setup parameters
clearvars;clc;
cd('/nfs/arch11/researchData/USER/tbruijne/Projects_Main/ReconSocket/recon-socket-repo/recon-socket/matlab_scripts/caspr/utils')
% run caspr_machine_setup.m
% par = check_default_values(par);
N = 64;
text_file_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/machine_flip_angles.txt';
data_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/img_data.h5';

% While loop which takes and proccesses an image
while 1    
    if ~exist(text_file_loc)
        % Generate random image
        img = rand(1)*ones(N,N) + 0.05*rand(N,N);

        % (if applicable remove locked file)
        if exist([data_loc,'.lck'])
            delete([data_loc,'.lck'])
        end
        % Create new (locked) datafile
        if ~exist(data_loc)
            h5create([data_loc,'.lck'],'/img',[N N])
        else
            system(['mv ',data_loc,' ',[data_loc,'.lck']])
        end
        % Write image to data file
        h5write([data_loc,'.lck'],'/img',img);
        % Unlock data file
        system(['mv ',[data_loc,'.lck'],' ',data_loc]);

        % Display info
        disp(['RMI: Passed image...']);

        % Wait for Python to respond
        while exist(data_loc)
            pause(0.1);
        end

    else
        disp(['RMI: waiting...']);pause(0.1);
    end
           
end
