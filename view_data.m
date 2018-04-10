% view_data displays the selected category/video from the CDnet2014
% dataset.
% set category and video variables.
clear all
close all
clc

datasetPath = '/datasets/backsub/cdnet2014/dataset';
category = 'turbulence';
video = 'turbulence3';

categoryList = filesys('getFolders', datasetPath);
categoryPath = fullfile(datasetPath, category);
videoList = filesys('getFolders', categoryPath);
videoPath = fullfile(categoryPath, video);
imagesPath = fullfile(videoPath, 'input');

imageFiles = filesys('getFiles', imagesPath);

imName = fullfile(imagesPath, imageFiles{1});
img = imread(imName);
h = imshow(img);

for i=1:length(imageFiles)
    imName = fullfile(imagesPath, imageFiles{i});
    img = imread(imName);
    set(h, 'CData', img)
    
    drawnow
    
    disp(i)
    %pause(0.001)
end

%background = estimateBackground(imagesPath, 1, 300);
%figure, imshow(background)

disp('Finish!')