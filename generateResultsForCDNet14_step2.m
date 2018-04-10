%% Run the CDNet14 metric calculations and save the results

% Metric calculation scripts
addpath('matlab stats')

datasetPath = '/datasets/backsub/cdnet2014';

% Run the scripts on the following number of passes
numPasses = [1:2];

for numPass = numPasses
    imageResultsDir = fullfile(datasetPath, sprintf('results_%d_passes', numPass));
    
    % Run the CDNet2014 metric calculation script
    processFolder(fullfile(datasetPath, 'dataset'), imageResultsDir, numPass)
end