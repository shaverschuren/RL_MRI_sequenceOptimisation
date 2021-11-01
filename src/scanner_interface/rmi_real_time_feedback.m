%% Reconsocket Matlab Interface to process images
% This script downloads the images on the RMQ exchange server using our mex
% interface.

% Some setup parameters
clearvars;clc;
cd('/nfs/arch11/researchData/USER/tbruijne/Projects_Main/ReconSocket/recon-socket-repo/recon-socket/matlab_scripts/caspr/utils')
% run caspr_machine_setup.m
% par = check_default_values(par);
% par.machine_id = "rtrabbit";
text_file_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/machine_flip_angles.txt';
img_file_loc = '/nfs/rtsan01/RT-Temp/TomBruijnen/img.h5';

% Establish connection with server
ReconSocketWrapper();
ReconSocketWrapper(par.machine_id);

% Create database (if applicable)
if ~exist(data_loc)
    h5create(data_loc,'/img')
end

% While loop which takes and proccesses an image

img_store = [];
while 1
    % Read current image
    img = ReconSocketWrapper(2);
    if ~isempty(img)
        img = reshape(img(1).real,img(1).size_dim');
        if isempty(img_store)
            img_store = img;
        else
            img_store(:,:,end+1) = img;
        end
        % Store image (ready for python RL model)
        h5write(data_loc,'/img',img_store);
    end

    disp(['RMI: waiting...']);pause(1);
           
end
