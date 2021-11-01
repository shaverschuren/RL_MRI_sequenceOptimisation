%% Reconsocket Matlab Interface to process images
% This script downloads the images on the RMQ exchange server using our mex
% interface.

% Some setup parameters
clearvars;clc;
cd('/nfs/arch11/researchData/USER/tbruijne/Projects_Main/ReconSocket/recon-socket-repo/recon-socket/matlab_scripts/caspr/utils')
run caspr_machine_setup.m
par = check_default_values(par);
par.machine_id = "rtrabbit";
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
        % Set up image
        img = reshape(img(1).real,img(1).size_dim');
        % Store image in img_store
        if isempty(img_store)
            img_store = img;
        else
            img_store(:,:,end+1) = img;
        end

        % Write image to data drop (for Python pickup)
        h5write(data_loc,'/img',img);
        % Create file to signal completion to Python
        fid = fopen(signal_file_loc, 'w');
        fclose(fid);
        disp(['RMI: Passed image...']);
        % Wait for Python to respond
        while exist(signal_file_loc)
            pause(0.1);
        end
    end

    disp(['RMI: waiting...']);pause(0.1);
           
end
