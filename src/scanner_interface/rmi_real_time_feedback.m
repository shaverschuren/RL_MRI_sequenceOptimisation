%% Reconsocket Matlab Interface to process images
% This script downloads the images on the RMQ exchange server using our mex
% interface.

% Setup file
clearvars;clc;

% Read config data
[script_dir, ~, ~] = fileparts(mfilename('fullpath'));
cd(script_dir)
json_id = fopen('../../config.json');
json_val = jsondecode(char(fread(json_id,inf)'));
fclose(json_id);

% Setup some parameters and utility files
cd(json_val.matlab_util_loc)
run caspr_machine_setup.m
par = check_default_values(par);
par.machine_id = "rtrabbit";
text_file_loc = json_val.param_loc;
data_loc = json_val.data_loc;

% Establish connection with server
ReconSocketWrapper();
ReconSocketWrapper(par.machine_id);

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

        % (if applicable remove locked data file)
        if exist([data_loc,'.lck'])
            delete([data_loc,'.lck']);
        end
        % Create new (locked) datafile
        if ~exist(data_loc)
            h5create([data_loc,'.lck'],'/img',[N N]);
        else
            system(['mv ',data_loc,' ',[data_loc,'.lck']]);
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
    end

    disp(['RMI: waiting...']);pause(0.1);
           
end
