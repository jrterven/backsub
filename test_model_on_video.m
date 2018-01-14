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
numPasses = 5;

% Trained model to use
trainedModel = '/datasets/backsub/checkpoints/model03_shadow.mat';

% Video file
videoPath = '/datasets/tracking/multi_cam/ICGLab6/ICGLab6/chap/cam131.avi';

% Background image
backgroundImagePath = '/datasets/tracking/multi_cam/ICGLab6/ICGLab6/chap/back_131.jpg';


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
subplot(3, 3, 1); h1 = imshow(im); title('Input Image');  % image
subplot(3, 3, 3); h2 = imshow(im); title('Background');  % background
subplot(3, 3, 2); h3 = imshow(im); title('Initial foreground'); % simple estimated foreground
subplot(3, 3, 4); h4 = imshow(im); title('Image for model'); % image for model 1
subplot(3, 3, 5); h5 = imshow(im); title('Result after 1 pass'); % result after 1 pass
subplot(3, 3, 6); h6 = imshow(im2); title('Results 1 pass overlay'); % result after 1 passes overlay
subplot(3, 3, 8); h7 = imshow(im);  title(sprintf('Results after %d passes', numPasses));% result after n passes
subplot(3, 3, 9); h8 = imshow(im2); title(sprintf('Results %d passes overlay', numPasses)); % result after n passes overlay

% Load model
data = load(trainedModel);
net = data.net;
    
% Read the background image
backgroundImg = imread(backgroundImagePath);
backgroundImg = rgb2gray(im2double(backgroundImg));
backgroundImg = imresize(backgroundImg,[img_h img_w]);

v = VideoReader(videoPath);

for i = 1:1000
    % Read one of the images and extract the channels
    img = readFrame(v);
    img = imresize(img,[img_h img_w]);
    img = rgb2gray(im2double(img));  

    % Get the foreground by simple background subtraction
    bgsub = img;
    backgroundImg = im2double(backgroundImg);
    bgsub(abs(img - backgroundImg) <= 0.1 ) = 0;
    bgsub(bgsub > 0) = 1;
    bgsub = imopen(bgsub, strel('rectangle', [3,3]));
    bgsub = imclose(bgsub, strel('rectangle', [10, 10]));
    bgsub = imfill(bgsub, 'holes');

    % images to u8
    img = im2uint8(img);
    backgroundImg = im2uint8(backgroundImg);
    bgsub = im2uint8(bgsub);
    
    % run model 1 time
    [imgForModel, fgCat, fg] = runModelNTimes(img, backgroundImg, bgsub, net, 1);
    overlayResult1 = labeloverlay(img, fgCat, 'Colormap', cmap, 'Transparency',0.4);
    
    % run model n times
    [~, fgCat, fg2] = runModelNTimes(img, backgroundImg, fg, net, 5);
    overlayResult2 = labeloverlay(img, fgCat, 'Colormap', cmap, 'Transparency',0.4);
    
    set(h1, 'CData', img)
    set(h2, 'CData', backgroundImg)
    set(h3, 'CData', bgsub)
    set(h4, 'CData', imgForModel)
    set(h5, 'CData', fg)
    set(h6, 'CData', overlayResult1)
    set(h7, 'CData', fg2)
    set(h8, 'CData', overlayResult2)
    
    drawnow
end

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
