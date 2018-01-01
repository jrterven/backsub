% view_data displays the selected category/video from the CDnet2014
% dataset.
% set category and video variables.
clear all
close all
clc

img_h = 240;
img_w = 320;
threshold = 0.1; % threshold for simple background subtraction

datasetPath = '/datasets/backsub/cdnet2014';
output_dir = fullfile(datasetPath, 'dataForTraining2');
output_imgs = fullfile(output_dir, 'images');
output_labels = fullfile(output_dir, 'groundtruth');
datasetPath = fullfile(datasetPath, 'dataset');

img = zeros(img_h, img_w, 3, 'double');
img2 = zeros(img_h, img_w, 1, 'double');
figure, 
subplot(2,3,1)
h1 = imshow(img2);  % background image
title('Background image')
subplot(2,3,2)
h2 = imshow(img);   % current image
title('Current frame')
subplot(2,3,3)
h3 = imshow(img2);  % simple foreground
title('Simple foreground')
subplot(2,3,4)
h4 = imshow(img);   % image for model
title('Image for training')
subplot(2,3,5)
h5 = imshow(img2);   % image for model
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

% Get the list of categories
categoryList = filesys('getFolders', datasetPath);

count_img = 0;
% For each category
for catIdx = 1:length(categoryList)
    category = categoryList{catIdx};
    categoryPath = fullfile(datasetPath, category);
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

        % Estimate the background
        fprintf('Estimating background of video %s-%s ...\n', video)
        backgroundImg = estimateBackground(imagesPath, 1, 300);
        backgroundImg = rgb2gray(im2double(backgroundImg));
        backgroundImg = imresize(backgroundImg,[img_h img_w]);
        set(h1, 'CData', backgroundImg);

        % for each frame on the video
        disp('Creating frames ...')
        for i = initFrame:finalFrame
            imName = fullfile(imagesPath, imageFiles{i});
            labelImgName = fullfile(labelsPath, labelFiles{i});

            img = imread(imName);
            img = imresize(img,[img_h img_w]);
            set(h2, 'CData', img);  

            labelImg = imread(labelImgName);
            labelImg = imresize(labelImg,[img_h img_w], 'nearest');
            set(h5, 'CData', labelImg);  

            img = rgb2gray(im2double(img));
            bgsub = img;

            % Subtract the background
            bgsub(abs(img - backgroundImg) <= threshold ) = 0;
            bgsub(bgsub > 0) = 1;
            bgsub = imopen(bgsub, strel('rectangle', [3,3]));
            bgsub = imclose(bgsub, strel('rectangle', [15, 15]));
            bgsub = imfill(bgsub, 'holes');

            set(h3, 'CData', bgsub); 

            % Create input data by stacking the 3 frames
            imgForModel = cat(3, img, backgroundImg, bgsub);
            set(h4, 'CData', imgForModel);

            drawnow;

            % Write image and label to disk
            imwrite(imgForModel,fullfile(output_imgs, sprintf('in%.6d.jpg', count_img)))
            imwrite(labelImg,fullfile(output_labels, sprintf('gt%.6d.png', count_img)))
            count_img = count_img + 1;
        end
    end
end

disp('Finish!');