function AMN_KArSL()
homeSourceFolder = '/home/eye/KArSLFrames/all/18';
fnames=rdir(strcat(homeSourceFolder, '/03/*/*/*'));
targetPath = '/home/eye/KArSLFrames/fusion/images/color/raw';

parfor i=1:length(fnames)
    %%
    fpath = fnames(i).name;
    [filePath, filename, ext] = fileparts(fpath);
    fpath
    newPath = replace(filePath, homeSourceFolder, targetPath);
    newFullPath = replace(fpath, homeSourceFolder, targetPath);
    
    %%%newFileName = fullfile(newPath, replace(filename,'.mp4',''));
    % extract frames
    filename
    %%%KeyFrameImagePath = replace(filePath, '/KArSL/','/KArSL/KeyFrames/KArSLImages/');
    sampleFolderFullPath = fpath; %%%fullfile(KeyFrameImagePath,replace(filename,'_c',''));
    keyFrameImages= rdir(strcat(sampleFolderFullPath,'/*.png'));
    %%%%vidReader = VideoReader(fpath);
    %/media/eye/B29A2B4F9A2B0F81/Hamzah/KArSL/KeyFrames/KArSLImages
    %AccumImgDifferences = [];
    targetFolder = newPath;%replace(sampleFolderFullPath,'KeyFrames/KArSLImages' ,'ImagesFused');
    
    if exist(strcat(newFullPath,'.jpg'),'file') ~= 2
        mkdir(newPath);
        f = 1;
        numFrames = length(keyFrameImages);
        if numFrames> 0
            while f <= numFrames
                
                imageFramePath = keyFrameImages(f).name;
                frameRGB_1 = imread(imageFramePath);
                if (f==1)
                    imgDiff = frameRGB_1;
                else
                    imgDiff = imfuse(imgDiff, frameRGB_1);
                end
                %imshow(imgDiff);
                f = f + 1;
            end
            
            %imshow(imgDiff);
            %I1 = read(vidReader, 1);
            %im = abs(imgDiff - I1);
            %imshow(im)
            imwrite(imgDiff,strcat(newFullPath,'.jpg'),'jpg','BitDepth', 8);
        end
    end
end

end