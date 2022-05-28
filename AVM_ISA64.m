function ImageFusing_v3()

% perform forward, backward, and conactenated fusion of the frames for ISA64 dataset where structure is [signer-->sign-->samples]
signers = {'001' '002' '003' '004' '005' '006' '007' '008' '009' '010'};
for s = 1: length(signers)
homeSourceFolder = strcat('/home/eye/lsa64_raw/images/' ,signers{s} );
homeSourceFolder
signs=rdir(strcat(homeSourceFolder, '/*/*'));
targetPath_forward = strcat('/home/eye/lsa64_raw/fusion/forward/',signers(s) );
targetPath_backward = strcat('/home/eye/lsa64_raw/fusion/backward/',signers(s) );
targetPath_both = strcat('/home/eye/lsa64_raw/fusion/both/',signers(s) );


for i=1: length(signs)
    %%
    fpath = signs(i).name;
    [filePath, filename, ext] = fileparts(fpath);
    fpath
    newPath_forward = replace(filePath, homeSourceFolder, targetPath_forward);
    newPath_backward = replace(filePath, homeSourceFolder, targetPath_backward);
    newPath_both = replace(filePath, homeSourceFolder, targetPath_both);
    
    
    % extract frames
    filename
    sampleFolderFullPath = fpath; %%%fullfile(KeyFrameImagePath,replace(filename,'_c',''));
    keyFrameImages= rdir(strcat(sampleFolderFullPath,'/*.jpg'));
    
    if exist(strcat(newPath_forward,'.jpg'),'file') ~= 2
        mkdir(newPath_forward);
        mkdir(newPath_backward);
        mkdir(newPath_both);
        
        numFrames = length(keyFrameImages);
        %%%% Backwarrd 
        f = numFrames;
        if numFrames> 0
            while f >0%<= numFrames
                
                imageFramePath = keyFrameImages(f).name;
                frameRGB_1 = imread(imageFramePath);
                if (f==numFrames)
                    imgDiff = frameRGB_1;
                else
                    imgDiff = imfuse(imgDiff, frameRGB_1);
                end
                %imshow(imgDiff);
                f = f - 1;
            end
            f = 1;
            %%%%%%%%% Forward %%%%%%%%%
            imgDiff_backward= imgDiff;
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
            imgDiff_forward= imgDiff;
            
            imgDiff_both = imfuse(imgDiff_backward, imgDiff_forward); 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %imshow(imgDiff);
            %I1 = read(vidReader, 1);
            %im = abs(imgDiff - I1);
            %imshow(im)
            disp(strcat(newPath_backward,'/',filename,'.jpg'));
            imwrite(imgDiff_backward,strcat(newPath_backward,'/',filename,'.jpg'));
            imwrite(imgDiff_forward,strcat(newPath_forward,'/',filename,'.jpg'));
            imwrite(imgDiff_both,strcat(newPath_both,'/',filename,'.jpg'));
            

        end
    end
end
end
end