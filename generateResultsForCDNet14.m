%% Generate image results for CDNet 2014 processFolder script
% CDNet 2014 provides a script to calculate the performance metrics called
% processFolder.m. This script assumes the results are contained in
% separate directories for each category. This script runs the models on
% on each dataset image and save the results inside the results directories.

clear all
close all
clc

gpuDevice(2)

% Images resolution for model
img_h = 240;
img_w = 320;

% Number of passes of the model
numPassesList = [1, 2];

% Use morphological operations
withMorp = [true, true, true, false, true, false, true, true, true, true, true];

% Threshold for simple background subtraction
threshold = [0 0.05 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]; 

% Dataset path
dataPath = '/datasets/backsub/cdnet2014';

trainedModel = '/datasets/backsub/checkpoints/modelcat3.mat';

% Video categories:
% 1:PTZ, 2:badWeather, 3:baseline, 4:camerajitter, 5:dynamicBackground,
% 6:intermittentObjectMotion, 7:lowFramerate, 8:nightVideos, 9:shadow,
% 10:thermal, 11:turbulence
categories = 3;

cmap = [
    1 0 0   % Background
    0 0 1   % Foreground
    ];

classes = [
    "Background"
    "Foreground"
    ];

hnd1 = figure;
set(0,'CurrentFigure',hnd1);
im = zeros(img_h, img_w, 'uint8');
im2 = zeros(img_h, img_w, 3, 'uint8');
subplot(2, 3, 1); h1 = imshow(im); title('Input Image');  % image
subplot(2, 3, 2); h3 = imshow(im); title('Initial foreground'); % simple estimated foreground
subplot(2, 3, 3); h2 = imshow(im); title('Background');  % background
subplot(2, 3, 4); h4 = imshow(im); title('Groundtruth'); % image for model 1
subplot(2, 3, 5); h5 = imshow(im);  title('Results');% result after n passes
subplot(2, 3, 6); h6 = imshow(im2); title('Results overlay'); % result after n passes overlay

% Loop for each category in categories list
for category = categories
    
    % Data paths
    datasetPath = fullfile(dataPath, 'dataset');
    categoryList = filesys('getFolders', datasetPath);
    categoryPath = fullfile(datasetPath, categoryList{category});
    videoList = filesys('getFolders', categoryPath);
    
    % Load model
    if isempty(trainedModel)
        trainedModel = fullfile('/datasets/backsub/checkpoints',['modelcat' num2str(category) '.mat']); 
    end
        
    fprintf('Loading model %s for category %s\n', trainedModel, categoryList{category});
    data = load(trainedModel);
    net = data.net;

    % Loop for each number of passes in numPassesList
    for passIdx = 1: length(numPassesList)
        numPasses = numPassesList(passIdx);
        fprintf('Running category: %s for %d passes\n', categoryList{category}, numPasses)
        
        % Results directory
        outputDir = fullfile(dataPath, sprintf('results_%d_passes',numPasses));

        if ~exist(outputDir,'dir')
            % Create a copy of results directory with the output_dir name
            copyfile(fullfile(dataPath, 'results'), outputDir);
        end

        % Each category contains several videos, 
        % for each video on the selected category
        for vIdx = 1:length(videoList)
            video = videoList{vIdx};
            
            fprintf('Running video: %s ...\n', video);
            
            videoPath = fullfile(categoryPath, video);
            imagesPath = fullfile(videoPath, 'input');
            labelsPath = fullfile(videoPath, 'groundtruth');
            imageFiles = filesys('getFiles', imagesPath);
            labelFiles = filesys('getFiles', labelsPath);

            % Get the temporal ROI
            tempROI = dlmread(fullfile(videoPath, 'temporalROI.txt'));
            initFrame = tempROI(1);
            finalFrame = tempROI(2);

            % Get the ROI mask (~0 = ROI)
            [roiImg, map] = imread(fullfile(videoPath, 'ROI.bmp'));
            if ~isempty(map)
                roiImg = ind2rgb(roiImg, map);
                if size(roiImg, 3) == 3
                    roiImg = roiImg(:, :, 1);
                end
            end

            % Get output image size
            out_h = size(roiImg, 1);
            out_w = size(roiImg, 2);

            % Get the background image
            backgroundImg = imread(fullfile(videoPath, 'background.jpg'));
            backgroundImg = rgb2gray(backgroundImg);
            backgroundImg(roiImg == 0) = 0;
            backgroundImg = imresize(backgroundImg,[img_h img_w]);

            % for each frame on the video
            for i = initFrame:finalFrame
                % Get the frame names
                imName = fullfile(imagesPath, imageFiles{i});
                labelImgName = fullfile(labelsPath, labelFiles{i});

                % Get the image, select the ROI and resize it
                img = imread(imName);
                img = rgb2gray(img);
                img(roiImg == 0) = 0;
                img = imresize(img,[img_h img_w]);
                img = im2double(img);    

                % Get the label and resize it
                labelImg = imread(labelImgName);
                labelImg = imresize(labelImg,[img_h img_w], 'nearest');

                % Get the foreground by simple background subtraction
                bgsub = img;
                backgroundImg = im2double(backgroundImg);
                bgsub(abs(img - backgroundImg) <= threshold(category)) = 0;
                bgsub(bgsub > 0) = 1;

                if withMorp(category)
                    bgsub = imopen(bgsub, strel('rectangle', [3,3]));
                    bgsub = imclose(bgsub, strel('rectangle', [10, 10]));
                    %bgsub = imdilate(bgsub, strel('rectangle', [14, 14]));
                    bgsub = imfill(bgsub, 'holes');
                end

                % images to u8
                img = im2uint8(img);
                backgroundImg = im2uint8(backgroundImg);
                bgsub = im2uint8(bgsub);

                % run model numPasses times
                [imgForModel, fgCat, fg] = runModelNTimes(img, backgroundImg, bgsub, net, numPasses);
                overlayResult1 = labeloverlay(img, fgCat, 'Colormap', cmap, 'Transparency',0.7);

                % Resize and save result on directory
                fgOut = imresize(fg,[out_h out_w],  'method', 'nearest');
                outDir = fullfile(outputDir, categoryList{category}, video);
                outImgName = fullfile(outDir, ['bin', num2str(i, '%.6d'), '.png']);
                imwrite(fgOut, outImgName);

                set(h1, 'CData', img)
                set(h2, 'CData', backgroundImg)
                set(h3, 'CData', bgsub)
                set(h4, 'CData', labelImg)
                set(h5, 'CData', fg)
                set(h6, 'CData', overlayResult1)
                drawnow
            end % for each frame
        end % for each video in category
    end % for each number of passes
end % for each category

disp('Finish!!')

function [imgForModel, fgCat, fg] = runModelNTimes(img, bg, fg, net, times)
% runModelNTimes run the ConvNet multiple times taking the previous output
% as input
%
%   Arguments:
%     img: input image (uint8 grayscale)
%     bg:  background image (uint8 grayscale)
%     fg:  estimated foreground (uint8 binary image 0=background, 255=foreground)
%     net: ConvNet
%     times: number of times to run the model
%          
%   Outputs:
%     imgForModel: final concatenated input image to model (for
%     visualization purposes)
%     fgCat: foreground image result categorical (uint8 1=background, 2=foreground)
%     fg: foreground image result (uint8: 0=background, 255=foreground)
    for i = 1 : times
        imgForModel = cat(3, img, bg, fg);
        fgCat = semanticseg(imgForModel, net);
        fgCat = uint8(fgCat);
        fg = fgCat;
        fg(fg == 1) = 0;
        fg(fg == 2) = 255;
    end
end
