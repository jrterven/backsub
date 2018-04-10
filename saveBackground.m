% view_data displays the selected category/video from the CDnet2014
% dataset.
% set category and video variables.
clear all
close all
clc

startFrame = 1;
numFrames = 1000;
useROI = false;

datasetPath = '/datasets/backsub/cdnet2014/dataset';
category = 'turbulence';
video = 'turbulence3';

categoryList = filesys('getFolders', datasetPath);
categoryPath = fullfile(datasetPath, category);
videoList = filesys('getFolders', categoryPath);
videoPath = fullfile(categoryPath, video);
imagesPath = fullfile(videoPath, 'input');

imageFiles = filesys('getFiles', imagesPath);

% Get the temporal ROI
tempROI = dlmread(fullfile(videoPath, 'temporalROI.txt'));
initFrame = tempROI(1);
finalFrame = tempROI(2);

imName = fullfile(imagesPath, imageFiles{1});
img = imread(imName);
h = imshow(img);

if useROI
    % Get the spatial ROI
    [roiImg, map] = imread(fullfile(videoPath, 'ROI2.bmp'));
    if ~isempty(map)
        roiImg = ind2rgb(roiImg, map);
        if size(roiImg, 3) == 3
            roiImg = roiImg(:, :, 1);
        end
    end

    % Get previous background image
    bg = imread(fullfile(videoPath, 'background.jpg'));
    
    background = estimateBackgroundReplaceROI(imagesPath, startFrame, ...
        numFrames, h, true, bg, roiImg);
else
    background = estimateBackground(imagesPath, startFrame, numFrames, h, true);
end


figure
imshow(background)
title('Estimated background');

imwrite(background, fullfile(videoPath, 'background.jpg'))

disp('Finish!')