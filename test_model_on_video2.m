%% Apply model on video file assuming an existing background image
% This model takes as input a background estimate, an input image and a
% background subtraction estimation.

clear all
close all
clc

% Images resolution for model
img_h = 240;
img_w = 320;

% Number of passes

% Trained model to use
trainedModel = '/datasets/backsub/checkpoints/model04_baselineLevels.mat';

% Video file
videoPath = '/datasets/aifi/video_office/cam1.avi';
outputVideoPath = '/datasets/aifi/video_office/cam1_results.avi';
frameRate = 5;

% Background image
backgroundImgOrg = estimateBackgroundFromVideo(videoPath, 160);
backgroundImgOrg = imresize(backgroundImgOrg,[img_h img_w]);

cmap = [
    1 0 0   % Background
    0 0 1       % Foreground
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
subplot(2, 3, 3); h2 = imshow(im); title('Background');  % background
subplot(2, 3, 2); h3 = imshow(im); title('Initial foreground'); % simple estimated foreground
subplot(2, 3, 4); h4 = imshow(im); title('Result after 1 pass'); % result after 1 pass
subplot(2, 3, 5); h5 = imshow(im); title('Results after 2 passes'); % result after 2 passes
subplot(2, 3, 6); h6 = imshow(im2); title('Results after 3 passes'); % result after 3 passes

% Load model
data = load(trainedModel);
net = data.net;
    
% Read the background image
backgroundImg = rgb2gray(im2double(backgroundImgOrg));

h = size(backgroundImg, 1) * 2;
w = size(backgroundImg, 2) * 3;
combinedImg = zeros(h, w, 3, 'uint8');
    
v = VideoReader(videoPath);
outputVideo = VideoWriter(outputVideoPath);
outputVideo.FrameRate = frameRate;
open(outputVideo)

i = 0;
while hasFrame(v)
    % Read one of the images and extract the channels
    img = readFrame(v);
    img = imresize(img,[img_h img_w]);
    imgGray = rgb2gray(im2double(img));  

    % Get the foreground by simple background subtraction
    bgsub = imgGray;
    backgroundImg = im2double(backgroundImg);
    bgsub(abs(imgGray - backgroundImg) <= 0.1 ) = 0;
    bgsub(bgsub > 0) = 1;
    bgsub = imopen(bgsub, strel('rectangle', [3,3]));
    %bgsub = imclose(bgsub, strel('rectangle', [10, 10]));
    bgsub = imdilate(bgsub, strel('rectangle', [14, 14]));
    bgsub = imfill(bgsub, 'holes');

    % images to u8
    imgGray = im2uint8(imgGray);
    backgroundImg = im2uint8(backgroundImg);
    bgsub = im2uint8(bgsub);
    
    % run model 1 time
    [imgForModel, fgCat, fg] = runModelNTimes(imgGray, backgroundImg, bgsub, net, 1);
    overlayResult1 = labeloverlay(imgGray, fgCat, 'Colormap', cmap, 'Transparency',0.7);
    
    % run model 2 times
    [~, fgCat, fg2] = runModelNTimes(imgGray, backgroundImg, fg, net, 2);
    overlayResult2 = labeloverlay(imgGray, fgCat, 'Colormap', cmap, 'Transparency',0.7);
    
    % run model 3 times
    [~, fgCat, fg3] = runModelNTimes(imgGray, backgroundImg, fg2, net, 3);
    overlayResult3 = labeloverlay(imgGray, fgCat, 'Colormap', cmap, 'Transparency',0.7);
    
    set(h1, 'CData', imgGray)
    set(h2, 'CData', backgroundImg)
    set(h3, 'CData', bgsub)
    set(h4, 'CData', overlayResult1)
    set(h5, 'CData', overlayResult2)
    set(h6, 'CData', overlayResult3)


    combinedImg(1:h/2, 1:w/3, :) = img;
    combinedImg(1:h/2, w/3+1:w*2/3, :) = backgroundImgOrg;
    combinedImg(1:h/2, w*2/3+1:w, :) = cat(3, bgsub, bgsub, bgsub);
    combinedImg(h/2+1:h, 1:w/3, :) = overlayResult1;
    combinedImg(h/2+1:h, w/3+1:w*2/3, :) = overlayResult2;
    combinedImg(h/2+1:h, w*2/3+1:w, :) = overlayResult3;
    writeVideo(outputVideo, combinedImg);
    
    drawnow
    
    pause(0.001)
    i = i + 1
end

close(outputVideo)
disp('Finish!')

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
