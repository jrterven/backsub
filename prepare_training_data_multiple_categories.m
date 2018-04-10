% TODO: remove un-labeled regions
% Set threshold depending on video
clear all
close all
clc

img_h = 240;
img_w = 320;
threshold = [0 0.05 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]; % threshold for simple background subtraction

dataPath = '/datasets/backsub/cdnet2014';
% Video categories:
% 1:PTZ, 2:badWeather, 3:baseline, 4:camerajitter, 5:dynamicBackground,
% 6:intermittentObjectMotion, 7:lowFramerate, 8:nightVideos, 9:shadow,
% 10:thermal, 11:turbulence

% Select categories to use
categories = [2 : 11];
numImagesForBackground = [500 100 100];
datasetPath = fullfile(dataPath, 'dataset');
categoryList = filesys('getFolders', datasetPath);

% Output directory
output_dir = fullfile(dataPath, 'dataForTrainingAll');
output_imgs = fullfile(output_dir, 'images');
output_labels = fullfile(output_dir, 'groundtruth');

img = zeros(img_h, img_w, 1, 'double');
figure, 
subplot(2,3,1)
h1 = imshow(img);  % background image
title('Background image')
subplot(2,3,2)
h2 = imshow(img);   % current image
title('Current frame')
subplot(2,3,3)
h3 = imshow(img);  % simple foreground
title('Simple foreground')
subplot(2,3,4)
h4 = imshow(img);   % image for model
title('Image for training')
subplot(2,3,5)
h5 = imshow(img);   % image for model
title('Label image')

% Create output directories, if already exists
if ~exist(output_dir,'dir')
    mkdir(output_dir)
    mkdir(output_imgs)
    mkdir(output_labels)
else
    rmdir(output_dir, 's')
    pause(1);
    mkdir(output_dir)
    mkdir(output_imgs)
    mkdir(output_labels)
end

count_img = 17316;
% For each category
for idx = 1:length(categories)
    catIdx = categories(idx);
    categoryName = categoryList{catIdx};
    categoryPath = fullfile(datasetPath, categoryName);
    videoList = filesys('getFolders', categoryPath);

    % For each video on the category
    for vIdx = 1:length(videoList)
        video = videoList{vIdx};
        videoPath = fullfile(categoryPath, video);
        imagesPath = fullfile(videoPath, 'input');
        labelsPath = fullfile(videoPath, 'groundtruth');
        imageFiles = filesys('getFiles', imagesPath);
        labelFiles = filesys('getFiles', labelsPath);

        % Get the temporal ROI
        tempROI = dlmread(fullfile(videoPath, 'temporalROI.txt'));
        initFrame = tempROI(1);
        finalFrame = tempROI(2);

        backgroundImg = imread(fullfile(videoPath, 'background.jpg'));
        backgroundImg = rgb2gray(im2double(backgroundImg));
        backgroundImg = imresize(backgroundImg,[img_h img_w]);

        % for each frame on the video
        disp('Creating frames ...')
        for i = initFrame:finalFrame
            % Get the frame names
            imName = fullfile(imagesPath, imageFiles{i});
            labelImgName = fullfile(labelsPath, labelFiles{i});

            % Get the image and resize it
            img = imread(imName);
            img = imresize(img,[img_h img_w]);
            img = rgb2gray(im2double(img));         

            % Get the label and resize it
            labelImg = imread(labelImgName);
            labelImg = imresize(labelImg,[img_h img_w], 'nearest');

            % Get the foreground by simple background subtraction
            bgsub = img;
            bgsub(abs(img - backgroundImg) <= threshold(catIdx) ) = 0;
            bgsub(bgsub > 0) = 1;
    %         bgsub = imopen(bgsub, strel('rectangle', [3,3]));
    %         bgsub = imclose(bgsub, strel('rectangle', [3, 3]));
    %         bgsub = imfill(bgsub, 'holes');

            % Zero-out the unlabeled regions
            img(labelImg == 85) = 0; 
            backgroundImg(labelImg == 85) = 0; 
            bgsub(labelImg == 85) = 0; 
            labelImg(labelImg == 85) = 0;

            % Create input data by stacking the 3 frames
            imgForModel = cat(3, img, backgroundImg, bgsub);

            % Display the frames
            set(h1, 'CData', backgroundImg);
            set(h2, 'CData', img);               
            set(h3, 'CData', bgsub); 
            set(h4, 'CData', imgForModel);
            set(h5, 'CData', labelImg);
            drawnow;

            % Write image and label to disk
            imwrite(imgForModel,fullfile(output_imgs, sprintf('in%.6d.jpg', count_img)))
            imwrite(labelImg,fullfile(output_labels, sprintf('gt%.6d.png', count_img)))
            count_img = count_img + 1;
        end
    end
end

disp('Finish!');