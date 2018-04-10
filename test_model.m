%% Apply model on video file
% This model takes as input a background estimate, an input image and a
% background subtraction estimation.

clear all
close all
clc

img_h = 240;
img_w = 320;

multiPass = 2;

% input trained model 
trainedModel = '/datasets/backsub/checkpoints/model04_baselineStage_1_2.mat';
videoPath = '/datasets/tracking/multi_cam/ICGLab6/ICGLab6/chap/cam131.avi';

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
subplot(3, 3, 1); h1 = imshow(im);  % image
subplot(3, 3, 2); h2 = imshow(im);  % background
subplot(3, 3, 3); h3 = imshow(im);  % simple estimated foreground
subplot(3, 3, 4); h4 = imshow(im);  % image for model 1
subplot(3, 3, 5); h5 = imshow(im);  % result after 1 pass
subplot(3, 3, 6); h6 = imshow(im2); % result after 1 passes overlay
subplot(3, 3, 7); h7 = imshow(im);  % image for model 2
subplot(3, 3, 8); h8 = imshow(im);  % result after 2 passes
subplot(3, 3, 9); h9 = imshow(im2); % result after 2 passes overlay

% Load model
data = load(trainedModel);
net = data.net;
    
% Read the background image
backgroundImg = imread('/datasets/tracking/multi_cam/ICGLab6/ICGLab6/chap/back_131.jpg');
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
    %bgsub = imclose(bgsub, strel('rectangle', [14, 14]));
    bgsub = imdilate(bgsub, strel('rectangle', [14, 14])); 
    bgsub = imfill(bgsub, 'holes');

    % images to u8
    img = im2uint8(img);
    backgroundImg = im2uint8(backgroundImg);
    bgsub = im2uint8(bgsub);
    
    % Image for model
    imgForModel1 = cat(3, img, backgroundImg, bgsub);

    % run model
    fgCat = semanticseg(imgForModel1, net);
    fgCat = uint8(fgCat);
    fg = fgCat;
    fg(fg == 1) = 0;
    fg(fg == 2) = 255;
    
    B1 = labeloverlay(img, fgCat, 'Colormap', cmap, 'Transparency',0.8);
    
    [fgCat, fg2] = runMultipleTimes(img, backgroundImg, fg, net, multiPass);
    
    B2 = labeloverlay(img, fgCat, 'Colormap', cmap, 'Transparency',0.8);
    
    set(h1, 'CData', img)
    set(h2, 'CData', backgroundImg)
    set(h3, 'CData', bgsub)
    set(h4, 'CData', imgForModel1)
    set(h5, 'CData', fg)
    set(h6, 'CData', B1)
    set(h8, 'CData', fg2)
    set(h9, 'CData', B2)
    
    drawnow
end

function [fgCat, fg] = runMultipleTimes(img, bg, fg, net, times)
    for i = 1 : times
        imgForModel = cat(3, img, bg, fg);
        fgCat = semanticseg(imgForModel, net);
        fgCat = uint8(fgCat);
        fg = fgCat;
        fg(fg == 1) = 0;
        fg(fg == 2) = 255;
    end
end
