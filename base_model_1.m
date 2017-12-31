%% Background subtraction Idea 1 on CDnet2014
% This model takes as input a background estimate, an input image and a
% background subtraction estimation.

clear all
close all
clc

%gpuDevice(2)

%vgg16();

% 1: train from scratch
% 2: resume training from checkpoint
% 3: do not train, just load model
doTraining = 1;

% input pre-trained model
pretrainedPath = '/home/juan/Dropbox/Projects/Background_subtraction/BackgroundSubtraction/code/backsub_model.mat';

% output trained model
trainedModelPath = '/home/juan/Dropbox/Projects/Background_subtraction/BackgroundSubtraction/code/backsub_model2.mat';

% last checkpoint
checkpoint = '/home/juan/Documents/checkpoints/convnet_checkpoint__37928__2017_12_31__11_41_31.mat';

%% CDnet2014 Dataset location
%outputFolder = '/data/datasets/cdnet2014/dataset/baseline/highway2';
outputFolder = '/datasets/backsub/cdnet2014/dataset/baseline/highway2';

% Estimate the background using the median of 100 images
bgImg = estimateBackground(fullfile(outputFolder, 'input'), 1, 100);
bgImg = rgb2gray(im2double(bgImg));

%% Load Images CDnet2014 Dataset
% Use |imageDatastore| to load CDnet2014 images. The |imageDatastore| enables you 
% to efficiently load a large collection of images on disk.
%%
imgDir = fullfile(outputFolder,'input');
imds = imageDatastore(imgDir);
%% 
% Display one of the images and the background
I = readimage(imds, 1);
I = rgb2gray(im2double(I));
bgsub = I;

% Subtract the background
th = 0.20; % threshold (bigger = remove more information)
bgsub(abs(I - bgImg) <= th ) = 0;
bgsub(bgsub > 0) = 1;

figure
subplot(1, 3, 1)
imshow(I)
% Display the background
subplot(1, 3, 2)
imshow(bgImg)

% Display background sub
subplot(1, 3, 3)
imshow(bgsub)

%% Load CDnet2014 Pixel-Labeled Images
% Use |pixelLabelDatastore| to load CDnet2014 pixel label image data. A
% |pixelLabelDatastore| encapsulates the pixel label data and the label ID
% to a class name mapping.
%
% Specify these classes.
%%
classes = [
    "Background"
    "Foreground"
    ];
%% 
% CDnet2014 annotations contains 5 labels:
% "Static", "Hard shadow", "Outside ROI", "Unknown motion", and "Motion"
% 
% The following function reduce these five clases into two: "Background"
% and "Foreground". See the function definition below for more details.

labelIDs = CDnet2014PixelLabelIDs();
%% 
% Use the classes and label IDs to create the |pixelLabelDatastore|:

labelDir = fullfile(outputFolder,'groundtruth');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
%% 
% Read and display one of the pixel-labeled images by overlaying it on top 
% of an image.

C = readimage(pxds, 1);

cmap = CDnet2014ColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);

figure
imshow(B)
pixelLabelColorbar(cmap,classes);
%% 
% Areas with no color overlay do not have pixel labels and are not used 
% during training.

%% Analyze Dataset Statistics
% To see the distribution of class labels in the CamVid dataset, use |countEachLabel|. 
% This function counts the number of pixels by class label.
%%
tbl = countEachLabel(pxds)
%% 
% Visualize the pixel counts by class.

frequency = tbl.PixelCount/sum(tbl.PixelCount);

figure
bar(1:numel(classes),frequency)
xticks(1:numel(classes)) 
xticklabels(tbl.Name)
xtickangle(45)
ylabel('Frequency')
%% 
% Ideally, all classes would have an equal number of observations. However,
% the classes in the CDnet2014 dataset are imbalanced, which is a common 
% issue in background subtraction datasets. Such scenes have more background 
% than foreground. If not handled correctly, this imbalance can be detrimental 
% to the learning process because the learning is biased in favor of the 
% dominant classes. Later on in this example, you will use class weighting to handle this issue.

%% Create input images for model
% Resize the images, convert to gray subtract the background and stack
% the gray input, the background, and the background subtraction.
%%
imagesSize = [240, 320];
disp('Preparing images offline ...')
imageFolder = fullfile(outputFolder,'imagesReszed',filesep);
imds = prepareImagesForModel(imds, imagesSize, bgImg, imageFolder, 0.2);

disp('Resizing labels ...')
labelFolder = fullfile(outputFolder,'labelsResized',filesep);
pxds = resizePixelLabels(pxds, imagesSize, labelFolder);

%% Prepare Training and Test Sets 
% The model is trained using 70% of the images from the dataset. The rest of the 
% images are used for testing. The following code randomly splits the image and 
% pixel label data into a training and test set.
%%
disp('Prepare training and test sets')
[imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionData(imds, pxds, 0.7);
%% 
% The 70/30 split results in the following number of training and test 
% images:

numTrainingImages = numel(imdsTrain.Files)
numTestingImages = numel(imdsTest.Files)

%% Create the Network
% As shown earlier, the classes in CamVid are not balanced. To improve
% training, you can use class weighting to balance the classes. Use the
% pixel label counts computed earlier with |countEachLabel| and calculate
% the median frequency class weights [1].

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

imageSize = [imagesSize 3];
numClasses = numel(classes);
%lgraph = segnetLayers(imageSize,numClasses,'vgg16');

lgraph = model01(imageSize, numClasses, tbl, classWeights);
%figure
%plot(lgraph)

%% Select Training Options
% The optimization algorithm used for training is stochastic gradient
% decent with momentum (SGDM). Use |trainingOptions| to specify the
% hyperparameters used for SGDM.

options = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-4, ...
    'L2Regularization', 0.0005, ...
    'MaxEpochs', 100, ...  
    'MiniBatchSize', 1, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 10, ...
    'Plots','training-progress', ...
    'CheckpointPath','/home/juan/Documents/checkpoints');
    
%% Data Augmentation
% Data augmentation is used during training to provide more examples to the
% network because it helps improve the accuracy of the network. Here,
% random left/right reflection and random X/Y translation of +/- 10 pixels
% is used for data augmentation. Use the |imageDataAugmenter| to specify
% these data augmentation parameters.

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation', [-10 10], 'RandYTranslation',[-10 10]);
%% 
% |imageDataAugmenter| supports several other types of data augmentation.
% Choosing among them requires empirical analysis and is another level of
% hyperparameter tuning.

%% Start Training
% Combine the training data and data augmentation selections using |pixelLabelImageSource|. 
% The |pixelLabelImageSource| reads batches of training data, applies data 
% augmentation, and sends the augmented data to the training algorithm.

datasource = pixelLabelImageSource(imdsTrain,pxdsTrain,...
    'DataAugmentation',augmenter);
%% 
% Startl training using |trainNetwork| if the |doTraining| flag is true. Otherwise, 
% load a pretrained network. Note: Training takes about 5 hours on an NVIDIA(TM) 
% Titan X and can take even longer depending on your GPU hardware.

if doTraining == 1          % train from scratch
    [net, info] = trainNetwork(datasource,lgraph,options);
    
    % save trained model
    save(trainedModelPath, 'net', 'info')
elseif doTraining == 2      % resume from checkpoint
    load(checkpoint)
    [net, info] = trainNetwork(datasource, net.Layers, options);
    
    % save trained model
    save(trainedModelPath, 'net', 'info')
    
elseif doTraining == 3      % do not train, load pre-trained model
    data = load(pretrainedPath);
    net = data.net;
end

%% Test Network on One Image
% As a quick sanity check, run the trained network on one test image. 
%%
I = read(imdsTest);
C = semanticseg(I, net);
%% 
% Display the results.

B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency',0.4);
figure
imshow(B)
pixelLabelColorbar(cmap, classes);
%% 
% Compare the results in |C| with the expected ground truth stored in |pxdsTest|. 
% The green and magenta regions highlight areas where the segmentation results 
% differ from the expected ground truth.

expectedResult = read(pxdsTest);
actual = uint8(C);
expected = uint8(expectedResult);
imshowpair(actual, expected)

%% 
% Visually, the semantic segmentation results overlap well for classes such 
% as road, sky, and building. However, smaller objects like pedestrians and cars 
% are not as accurate. The amount of overlap per class can be measured using the 
% intersection-over-union (IoU) metric, also known as the Jaccard index. Use the 
% |jaccard| function to measure IoU.

iou = jaccard(C, expectedResult);
table(classes,iou)
%% 
% The IoU metric confirms the visual results. Road, sky, and building classes 
% have high IoU scores, while classes such as pedestrian and car have low scores. 
% Other common segmentation metrics include the <matlab:doc('dice') Dice index> 
% and the <matlab:doc('bfscore') Boundary-F1> contour matching score.
%% Evaluate Trained Network 
% To measure accuracy for multiple test images, run |semanticseg| on the entire 
% test set. 
%%
pxdsResults = semanticseg(imdsTest,net,'WriteLocation',tempdir,'Verbose',false);
%% 
% |semanticseg| returns the results for the test set as a |pixelLabelDatastore| 
% object. The actual pixel label data for each test image in |imdsTest| is written 
% to disk in the location specified by the |'WriteLocation'| parameter. Use |evaluateSemanticSegmentation| 
% to measure semantic segmentation metrics on the test set results. 

metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',false);

%% 
% |evaluateSemanticSegmentation| returns various metrics for the entire 
% dataset, for individual classes, and for each test image. To see the dataset 
% level metrics, inspect |metrics.DataSetMetrics| .

metrics.DataSetMetrics
%% 
% The dataset metrics provide a high-level overview of the network performance. 
% To see the impact each class has on the overall performance, inspect the per-class 
% metrics using |metrics.ClassMetrics|.

metrics.ClassMetrics
%% 
% Although the overall dataset performance is quite high, the class metrics 
% show that underrepresented classes such as |Pedestrian|, |Bicyclist|, and |Car|
% are not segmented as well as classes such as |Road|, |Sky|, and |Building| . 
% Additional data that includes more samples of the underrepresented classes might 
% help improve the results.
%% References
% [1] Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. "SegNet: 
% A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." _arXiv 
% preprint arXiv:1511.00561_, 2015.
%
% [2] Brostow, Gabriel J., Julien Fauqueur, and Roberto Cipolla. "Semantic object 
% classes in video: A high-definition ground truth database." _Pattern Recognition 
% Letters_ Vol 30, Issue 2, 2009, pp 88-97.
% 
%% Supporting Functions
%%
function labelIDs = CDnet2014PixelLabelIDs()
% Return the label IDs corresponding to each class.
%
% CDnet2014 annotations are the following:
% 0   : Static
% 50  : Hard shadow
% 85  : Outside region of interest
% 170 : Unknown motion (usually around moving objects, due to semi-transparency and motion blur)
% 255 : Motion
%
% We will reduce these five clases into two such that:
% "Static", "Hard shadow", and "Uknown motion" are combined into "Background"
% and "Motion" is the "Foreground"
% "Outside region of interest" is ignored and not used during training
labelIDs = { ...
    
    % "Background"
    [
    0; ... % "Static"
    50; ...% "Hard shadow"
    170; ... % "Unknown motion"
    85; ... % "Outside ROI"
    ]
    
    % "Foreground" 
    [
    255; ... % "Motion"
    ]
    };
end
%% 
% 

function pixelLabelColorbar(cmap, classNames)
% Add a colorbar to the current axis. The colorbar is formatted
% to display the class names with the color.

colormap(gca,cmap)

% Add colorbar to current figure.
c = colorbar('peer', gca);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end
%% 
% 

function cmap = CDnet2014ColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    255 0 0   % Background
    0 0 255       % Foreground
    ];

% Normalize between [0 1].
cmap = cmap ./ 255;
end
%% 
% 

function imds = prepareImagesForModel(imds, newSize, backgroundImg, imageFolder, threshold)
if ~exist(imageFolder,'dir') 
    mkdir(imageFolder)
else
    imds = imageDatastore(imageFolder);
    return; % Skip if images already resized
end

reset(imds)
while hasdata(imds)
    % Read an image.
    [I,info] = read(imds);     
    
    % Resize image.
    I = imresize(I,newSize);    
    
    I = rgb2gray(im2double(I));
    bgsub = I;

    % Subtract the background
    bgsub(abs(I - backgroundImg) <= threshold ) = 0;
    bgsub(bgsub > 0) = 1;
    
    % Create input data by stacking the 3 frames
    I = cat(3, I, backgroundImg, bgsub);
    
    % Write to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(I,[imageFolder filename ext])
end

imds = imageDatastore(imageFolder);
end
%% 
% 

function pxds = resizePixelLabels(pxds, newSize, labelFolder)
% Resize pixel label data to newSize

classes = pxds.ClassNames;
labelIDs = 1:numel(classes);
if ~exist(labelFolder,'dir')
    mkdir(labelFolder)
else
    pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
    return; % Skip if images already resized
end

reset(pxds)
while hasdata(pxds)
    % Read the pixel data.
    [C,info] = read(pxds);
    
    % Convert from categorical to uint8.
    L = uint8(C);
    
    % Resize the data. Use 'nearest' interpolation to
    % preserve label IDs.
    L = imresize(L,newSize,'nearest');
    
    % Write the data to disk.
    [~, filename, ext] = fileparts(info.Filename);
    imwrite(L,[labelFolder filename ext])
end

labelIDs = 1:numel(classes);
pxds = pixelLabelDatastore(labelFolder,classes,labelIDs);
end
%% 
% 

function [imdsTrain, imdsTest, pxdsTrain, pxdsTest] = partitionData(imds, pxds, trainPercentage)
% Partition CamVid data by randomly selecting 60% of the data for training. The
% rest is used for testing.
    
% Set initial random state for example reproducibility.
rng(0); 
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use trainPercentage of the images for training.
N = round(trainPercentage * numFiles);
trainingIdx = shuffledIndices(1:N);

% Use the rest for testing.
testIdx = shuffledIndices(N+1:end);

% Create image datastores for training and test.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;
labelIDs = 1:numel(pxds.ClassNames);

% Create pixel label datastores for training and test.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end