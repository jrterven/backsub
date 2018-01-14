
clear all
close all
clc

img_h = 240;
img_w = 320;
frameRate = 20;

% input trained model 
trainedModel = '/datasets/backsub/checkpoints/model03_shadow.mat';

videoPath = '/datasets/tracking/multi_cam/ICGLab6/ICGLab6/leaf1/cam131.avi';
backgroundImgPath = '/datasets/tracking/multi_cam/ICGLab6/ICGLab6/leaf1/bg1.jpg';

outputVideoPath = '/datasets/tracking/multi_cam/ICGLab6/ICGLab6/leaf1/fgcam131.avi';

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
subplot(2, 3, 1); h1 = imshow(im);  % image
subplot(2, 3, 2); h2 = imshow(im);  % background
subplot(2, 3, 3); h3 = imshow(im);  % simple estimated foreground
subplot(2, 3, 4); h4 = imshow(im);  % image for model 1
subplot(2, 3, 5); h5 = imshow(im);  % result after 1 pass
subplot(2, 3, 6); h6 = imshow(im2); % result after 1 passes overlay


% Load model
disp('Loading model')
data = load(trainedModel);
net = data.net;
    
% Read the background image
backgroundImg = imread(backgroundImgPath);
backgroundImg = rgb2gray(im2double(backgroundImg));
backgroundImg = imresize(backgroundImg,[img_h img_w]);

% Create output video
disp('Creating output video')
outputVideo = VideoWriter(outputVideoPath, 'Grayscale AVI');
outputVideo.FrameRate = frameRate;
open(outputVideo)

v = VideoReader(videoPath);

while hasFrame(v)
    % Read one of the images and extract the channels
    img = readFrame(v);
    imgHeight = size(img, 1);
    imgWidth = size(img, 2);
    
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
    
    % Image for model
    imgForModel1 = cat(3, img, backgroundImg, bgsub);

    % run model
    fgCat = semanticseg(imgForModel1, net);
    fgCat = uint8(fgCat);
    fg = fgCat;
    fg(fg == 1) = 0;
    fg(fg == 2) = 255;
    
    B1 = labeloverlay(img, fgCat, 'Colormap', cmap, 'Transparency',0.4);
    
    set(h1, 'CData', img)
    set(h2, 'CData', backgroundImg)
    set(h3, 'CData', bgsub)
    set(h4, 'CData', imgForModel1)
    set(h5, 'CData', fg)
    set(h6, 'CData', B1)
    
    drawnow
    
    % Resize image to original size
    bgSubBig = imresize(fg, [imgHeight imgWidth], 'method', 'nearest');
    
    writeVideo(outputVideo, bgSubBig);
end

close(outputVideo)
disp('Finish!')

