%% Reconsocket Matlab Interface to process images
% This script simulates the behaviour of the scanner for
% prototyping purposes.

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
% run caspr_machine_setup.m
% par = check_default_values(par);
N = 64;
text_file_loc = json_val.param_loc;
data_loc = json_val.data_loc;

% While loop which takes and proccesses an image
while 1    
    if ~exist(text_file_loc)
        % Generate random image
        img = rand(1)*ones(N,N) + 0.05*rand(N,N);

        % (if applicable remove locked file)
        if exist([data_loc,'.lck'])
            delete([data_loc,'.lck']);
        end
        % Create new (locked) datafile
        if ~exist(data_loc)
            h5create([data_loc,'.lck'],'/img',size(img));
        else
            system(['mv ',data_loc,' ',[data_loc,'.lck']]);
        end
        % Write image to data file
        h5write([data_loc,'.lck'],'/img',rescale(img));
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
