import os
import fnmatch
import argparse
import numpy as np
import make_frames

# This code designed for converting videos into frames for ISA64 dataset where structure is [signer-->sign-->sample]

def mkfp(*dirs):
    return os.path.join(*dirs)

def get_filename(filename, ext):
    parts = filename.split(os.path.sep)
    return parts[-1].replace('.jpg', '')

def extractFrame(cat_odir, newCatPath, video_ext, ext, data_path, sign, cat,total_num, catWithinSign):
    
    for vid in fnmatch.filter(os.listdir(cat_odir), video_ext):
        
        if catWithinSign:
            catPath = os.path.join(data_path,sign, cat)
        else:
            catPath = os.path.join(data_path, cat,sign)
            
        videoOrMat = 'video'
        if vid.endswith('mat'):
            #ext ='.mat'
            videoOrMat = 'mat'
            if  vid.endswith('_d.mat'):
                skeleton_file =  mkfp(catPath, vid.replace('d.mat','c_s.mat'))
            else:
                skeleton_file =  mkfp(catPath, vid.replace('c.mat','c_s.mat'))
                
        else:
            #ext ='.mp4'
            
            if  vid.endswith('_d.mp4'):
                skeleton_file =  mkfp(catPath, vid.replace('d.mp4','c_s.mat'))
            else:
                skeleton_file =  mkfp(catPath, vid.replace('c.mp4','c_s.mat'))
                
        #print(catPath)    
        filename = vid.replace(ext,'')
        try:
            print('----')
            vpath = os.path.join(catPath, vid)
            odir = os.path.join(newCatPath, filename)
            #print(odir)

            if not os.path.isdir(odir):
                os.mkdir(odir)
    
            print("Decoding '%s' '%s'" % (vpath, odir))
            num_frames,fr =  make_frames.cv2_dump_frames(total_num, vpath, odir, skeleton_file, videoOrMat, image_type, segmentImage, resize, imWidth, imHeight, frameFormat, filename, 94)
            
            total_num = total_num+1
            #total_frames += num_frames
        
        except:
            print('Error :'+vpath)
            errFile.write(vpath + "\n")
            
    return total_num

total_frames = 0
fr_t = []
total_num = 1
video_ext = '*.mp4' #'*c.mp4' or '*d.mp4'
ext ='.avi'  
frameFormat ='jpg'
image_type = 'color'#'depth'
imWidth=256
imHeight=256
segmentImage =False # depth: False
catWithinSign=False # True: if Train, test, val folders are within sign
resize = True
data_path_src = '/home/eye/lsa64_raw/divided/'  
outDir_out = '/home/eye/lsa64_raw/images/'  

signers = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
fileCount = 0
for signerID in signers:
    data_path = os.path.join(data_path_src, signerID)
    outDir = os.path.join(outDir_out, signerID)
    if 	os.path.exists(outDir)==False:
        os.mkdir(outDir)

    errorFile ='error.txt'
    errFile = open(errorFile,'a')
    all_signs =sorted(os.listdir(data_path),reverse=True) #sorted(os.listdir(data_path),reverse=True)

    if catWithinSign == True:
        for sign in all_signs:
            print(sign)
            sign_Folder = os.path.join(data_path, sign)
            
            for cat in sorted(os.listdir(sign_Folder)):
                cat_odir = os.path.join(sign_Folder, cat)
                newCatPath = os.path.join(outDir, cat, sign)
                
                if not os.path.isdir(newCatPath):
                    os.makedirs(newCatPath)
                total_num = extractFrame(cat_odir, newCatPath, video_ext, ext, data_path, sign, cat,total_num, catWithinSign)
        
        
    else:
        for signParent in sorted(os.listdir(data_path)):
            
            signIDFolder = os.path.join(data_path, signParent)
            signCat_out = os.path.join(outDir, signParent)

            if os.path.exists(signCat_out)  == False:
                os.mkdir(signCat_out,0o755)

            

            for sign in sorted(os.listdir(signIDFolder)):
                print(signIDFolder +"/" + sign)
                signLabel = sign.split('.')[0]
                sign_path = os.path.join(signIDFolder, sign)
                sign_Folder_out = os.path.join(signCat_out, signLabel)  
                
                if os.path.exists(sign_Folder_out)  == False:
                    os.mkdir(sign_Folder_out,0o755)
                            
                num_frames,fr =  make_frames.cv2_dump_frames(total_num, sign_path, sign_Folder_out, imWidth=256, imHeight=256, fmt="jpg", filename='', quality=90, resizeImg=True, videoOrMat='video', image_type='color')
                total_num = total_num + 1
        
 
errFile.close()
print("Total frames decoded: %d" % total_frames)



