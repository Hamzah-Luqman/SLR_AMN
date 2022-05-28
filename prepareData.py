"""
This code has been used with some modifications from 
https://github.com/FrederikSchorr/sign-language

Assume videos are stored in sVideoDir as: 
... sVideoDir / train / class001 / gesture.mp4
... sVideoDir / val   / class249 / gesture.avi

This pipeline
* extracts frames/images from videos (saved in sVideoDir path)  and save frames in sImageDir folder
* Extract MobileNet features from frames  and save them in diFeature folder 
"""

import os
from frame import videosDir2framesDir
from feature import features_2D_load_model, features_2D_predict_generator


def startPipiline(diVideoSet, sVideoDir, diFeature, sImageDir, sImageFeatureDir):
    # extract frames from videos
    videosDir2framesDir(sVideoDir, sImageDir, nFramesNorm=diVideoSet["nFramesNorm"],
                        nResizeMinDim=diVideoSet["nMinDim"], tuCropShape=diFeature["tuInputShape"][0:2])

    # Load pretrained MobileNet model without top layer
    keModel = features_2D_load_model(diFeature)

    # calculate MobileNet features from rgb frames
    features_2D_predict_generator(
        sImageDir + "/test",   sImageFeatureDir + "/test",   keModel, diVideoSet["nFramesNorm"])
    features_2D_predict_generator(
        sImageDir + "/train", sImageFeatureDir + "/train", keModel, diVideoSet["nFramesNorm"])

    # the end


# dataset
diVideoSet = {"sName": "KArSL",
              "nClasses": 190,   # number of classes
              "nFramesNorm": 18,    # number of frames per video
              "nMinDim": 224,   # smaller dimension of saved video-frames
              "tuShape": (224, 224),  # height, width
              "nFpsAvg": 30,
              "nFramesAvg": 18,
              "fDurationAvg": 4.0}  # seconds


# feature extractor
diFeature = {"sName": "mobilenet",
             "tuInputShape": (224, 224, 3),
             "tuOutputShape": (1024, )}
# diFeature = {"sName" : "Xception",
#    "tuInputShape" : (299, 299, 3),
#    "nOutput" : 2048,
#    "tuOutputShape" : (2048, )}

# diFeature = {"sName" : "vgg16",
#        "tuInputShape" : (224, 224, 3),
#       "nOutput" : 512,
#       "tuOutputShape" : (512, )}


#################

sVideoDir = './dataset/KArSL/videos/25/01'  # path of video files
sImageDir = './dataset/KArSL/images/25/01'  # path of image files
# Path to save extracted features
sImageFeatureDir = './dataset/KArSL/features/mobilenet_temp/25/images/01'

print("Extracting frames, optical flow and .......... features ...")
print(os.getcwd())

startPipiline(diVideoSet, sVideoDir, diFeature, sImageDir, sImageFeatureDir)
