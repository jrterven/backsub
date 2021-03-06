% estimateMedianBackground estimates the background using the median of n
% images.
%
%   Arguments:
%     videoPath: absolute path of the video
%     n: number images used for background estimation
%
%   Output:
%     background: [height, width, 3] image of the background.
function background = estimateBackgroundFromVideo(videoPath, n)
    
    v = VideoReader(videoPath);
    
    % extract the RGB channels of the first image
    img = readFrame(v);
    reds = img(:,:,1); % Red channel
    greens = img(:,:,2); % Green channel
    blues = img(:,:,3); % Blue channel
    
    % extract the channels of each image and stack the together
    % on the third dimension
    for i= 1 : n - 1
        img = readFrame(v);
        red = img(:,:,1); % Red channel
        green = img(:,:,2); % Green channel
        blue = img(:,:,3); % Blue channel
        
        % stack the channels on the third dimension
        reds = cat(3, reds, red);
        greens = cat(3, greens, green);
        blues = cat(3, blues, blue);
    end
    
    % calculate the median across third dimension
    rmed = median(reds, 3);
    gmed = median(greens, 3);
    bmed = median(blues, 3);
    
    % generate output image
    background = cat(3, rmed, gmed, bmed);
end