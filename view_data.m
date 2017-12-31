% view_data displays the selected category/video from the CDnet2014
% dataset.
% set category and video variables.
clear all
close all
clc

datasetPath = '/data/datasets/cdnet2014/dataset';
category = 'baseline';
video = 'pedestrians';

categoryList = filesys('getFolders', datasetPath);
categoryPath = fullfile(datasetPath, category);
videoList = filesys('getFolders', categoryPath);
videoPath = fullfile(categoryPath, video);
imagesPath = fullfile(videoPath, 'input');

imageFiles = filesys('getFiles', imagesPath);

h = figure;
ax = gca;

% for i=1:length(imageFiles)
%     imName = fullfile(imagesPath, imageFiles{i});
%     img = imread(imName);
%     image(ax, img)
%     
%     drawnow
%     %pause(0.001)
% end

background = estimateBackground(imagesPath, 1, 300);
figure, imshow(background)