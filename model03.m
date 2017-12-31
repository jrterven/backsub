function lgraph = model03(inputSize, numClasses, tbl, classWeights)
% Small model with single 8x downsampling, upsampling and 64 filters
% This models uses a DAG

%% Final Layer
% The final set of layers are responsible for making pixel classifications. 
% These final layers process an input that has the same spatial dimensions (height 
% and width) as the input image. However, the number of channels (third dimension) 
% is larger and is equal to number of filters in the last transposed convolution 
% layer. This third dimension needs to be squeezed down to the number of classes 
% we wish to segment. This can be done using a 1-by-1 convolution layer whose 
% number of filters equal the number of classes, e.g. 3.
finalLayers = [
    convolution2dLayer(1, numClasses, 'Name','conv1x1');
    softmaxLayer('Name','softmax')
    pixelClassificationLayer('Name','labels','ClassNames', tbl.Name, 'ClassWeights', classWeights)
    ];

%% Create the network
% Stack all the layers to complete the semantic segmentation network. 
net = [
    imageInputLayer(inputSize, 'Name', 'input');    
    downSamplingModule(3, 64, 1)
    upSamplingModule(64, 1)
    finalLayers
    ];

lgraph = layerGraph(net);

%% Downsampling module
% Stack the convolution, ReLU, and max pooling layers to create a network 
% that downsamples its input by a factor of 8.
function layers = downSamplingModule(filterSize, numFilters, stackNum)
    layers = [
        convolution2dLayer(filterSize, numFilters,'Padding','same','Name',sprintf('conv_%d_1', stackNum))
        batchNormalizationLayer('Name', sprintf('bn_%d_1', stackNum))
        reluLayer('Name', sprintf('relu_%d_1', stackNum))
        maxPooling2dLayer(2,'Stride',2,'Name', sprintf('pool_%d_1', stackNum));
        convolution2dLayer(filterSize, numFilters * 2,'Padding','same','Name',sprintf('conv_%d_2', stackNum))
        batchNormalizationLayer('Name', sprintf('bn_%d_2', stackNum))
        reluLayer('Name', sprintf('relu_%d_2', stackNum))
        maxPooling2dLayer(2,'Stride',2,'Name', sprintf('pool_%d_2', stackNum));
        convolution2dLayer(filterSize, numFilters * 2,'Padding','same','Name',sprintf('conv_%d_3', stackNum))
        batchNormalizationLayer('Name', sprintf('bn_%d_3', stackNum))
        reluLayer('Name', sprintf('relu_%d_3', stackNum))
        maxPooling2dLayer(2,'Stride',2,'Name', sprintf('pool_%d_3', stackNum));
        convolution2dLayer(filterSize, numFilters * 4,'Padding','same','Name',sprintf('conv_%d_4', stackNum))
        batchNormalizationLayer('Name', sprintf('bn_%d_4', stackNum))
        reluLayer('Name', sprintf('relu_%d_4', stackNum))
        maxPooling2dLayer(2,'Stride',2,'Name', sprintf('pool_%d_4', stackNum));
        ];

%% Upsampling Module
% The upsampling is done using the tranposed convolution layer (also commonly 
% referred to as "deconv" or "deconvolution" layer). When a transposed convolution 
% is used for upsampling, it performs the upsampling and the filtering at the 
% same time.
% 
% Create a transposed convolution layer to upsample by 2. 
% The 'Cropping' parameter is set to 1 to make the output size equal twice 
% the input size. 
% 
% Stack the transposed convolution and relu layers. An input to this set 
% of layers is upsampled by 8.

function layers = upSamplingModule(numFilters, stackNum)
    layers = [
        transposedConv2dLayer(4, numFilters, 'Stride', 2, 'Cropping',1, 'Name', sprintf('deconv_%d_1', stackNum));
        batchNormalizationLayer('Name', sprintf('bnd_%d_1', stackNum))
        reluLayer('Name', sprintf('relud_%d_1', stackNum))        
        transposedConv2dLayer(4, numFilters, 'Stride', 2, 'Cropping',1, 'Name', sprintf('deconv_%d_2', stackNum));
        batchNormalizationLayer('Name', sprintf('bnd_%d_2', stackNum))
        reluLayer('Name', sprintf('relud_%d_2', stackNum))        
        transposedConv2dLayer(4, numFilters, 'Stride', 2, 'Cropping',1, 'Name', sprintf('deconv_%d_3', stackNum));
        batchNormalizationLayer('Name', sprintf('bnd_%d_3', stackNum))
        reluLayer('Name', sprintf('relud_%d_3', stackNum))
        transposedConv2dLayer(4, numFilters, 'Stride', 2, 'Cropping',1, 'Name', sprintf('deconv_%d_4', stackNum));
        batchNormalizationLayer('Name', sprintf('bnd_%d_4', stackNum))
        reluLayer('Name', sprintf('relud_%d_4', stackNum))
        ];

function layer = iCreateConvLayer(filterSize, numFilters, name)

    layer = convolution2dLayer(filterSize, numFilters, ...
        'Padding', 'same', 'Name', name);

    % Effectively disable bias by setting learning rates to 0 and initializing
    % the bias to zero.
    layer.BiasLearnRateFactor = 0;
    layer.BiasL2Factor = 0;
    layer.Bias = zeros(1,1,numFilters, 'single');

    % Initialize weights using MSRA weight initialization:
    %   He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    %   human-level performance on imagenet classification." Proceedings of the
    %   IEEE international conference on computer vision. 2015.
    shape = [layer.FilterSize layer.NumChannels layer.NumFilters];
    n = prod([layer.FilterSize layer.NumChannels]);
    layer.Weights = sqrt(2/n) * randn(shape, 'single');
