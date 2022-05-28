"""
https://github.com/FrederikSchorr/sign-language

In some video classification NN architectures it may be necessary to calculate features 
from the (video) frames, that are afterwards used for NN training.

Eg in the MobileNet-LSTM architecture, the video frames are first fed into the MobileNet
and the resulting 1024 **features** saved to disc.
"""

import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers

from keras.layers import Dense

from datagenerator import FramesGenerator
from keras.layers import Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import (Conv1D, Conv2D, MaxPooling2D)


def features_2D_load_model(diFeature:dict) -> keras.Model:
    
    sModelName = diFeature["sName"]
    print("Load 2D extraction model %s ..." % sModelName)

    # load pretrained keras models
    if sModelName == "mobilenet":
        '''
        from tensorflow.keras.applications.mobilenet import preprocess_input
        input_img = tf.keras.layers.Input(shape=(224, 224, 3))
        input_img = preprocess_input(input_img)

        # load base model with top
        keBaseModel = tf.keras.applications.MobileNet(
            weights="imagenet",
            input_tensor=input_img,
            include_top = False)
        #print(keBaseModel.summary())
        # We'll extract features at the final pool layer
        cnn_out = keras.layers.GlobalAveragePooling2D()(keBaseModel.output)
        cnn_out = keras.layers.Dropout(0.6)(cnn_out)
        cnn_out = keras.layers.Dropout(0.6)(cnn_out)
        keModel = tf.keras.models.Model(keBaseModel.input, cnn_out)
        '''
        '''

        keBaseModel = tf.keras.applications.mobilenet.MobileNet(
            weights="imagenet",
            input_shape = (224, 224, 3),
            include_top = True)
        print(keBaseModel.summary())

        # We'll extract features at the final pool layer
        keModel = tf.keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.get_layer('global_average_pooling2d').output) 
        '''
        keBaseModel = tf.keras.applications.mobilenet.MobileNet(
            weights="imagenet",
            input_shape = (224, 224, 3),
            include_top = True)
         
        cnn_out = keBaseModel.get_layer('global_average_pooling2d').output
        cnn_out = keras.layers.Dropout(0.6)(cnn_out)
        cnn_out = keras.layers.Dropout(0.6)(cnn_out)
        keModel = tf.keras.models.Model(keBaseModel.input, cnn_out)


        print(keModel.summary())
 

    elif sModelName == "inception":

        # load base model with top
        keBaseModel = keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=True)

        # We'll extract features at the final pool layer
        keModel = keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.get_layer('avg_pool').output,
            name=sModelName + " without top layer") 
			
    elif sModelName == "vgg16":

        # load base model with top
        keBaseModel = VGG16(weights="imagenet",  input_shape=(224, 224, 3), include_top=False, pooling="avg")
		
        keModel = keras.models.Model(inputs=keBaseModel.input, outputs=keBaseModel.get_layer('global_average_pooling2d_1').output, name= "VGG 16 without top layer") 

    elif sModelName == "ResNet50":

        # load base model with top
        #keBaseModel = ResNet50(weights="imagenet",  input_shape=(224, 224, 3), include_top=False, pooling="avg")
		
        #keModel = keras.models.Model(inputs=keBaseModel.input, outputs=keBaseModel.get_layer('global_average_pooling2d_1').output, name= "ResNet50 without top layer") 
        keBaseModel = ResNet50(weights="imagenet",  input_shape=(224, 224, 3), include_top=True)
        print(keBaseModel.summary())
        keModel = keras.models.Model(inputs=keBaseModel.input, outputs=keBaseModel.get_layer('avg_pool').output, name= "ResNet50 without top layer") 
		
    elif sModelName == "Xception":

        # load base model with top
        keBaseModel = Xception(weights="imagenet",  input_shape=(299, 299, 3), include_top=True)
        print(keBaseModel.summary())
        keModel = keras.models.Model(inputs=keBaseModel.input, outputs=keBaseModel.get_layer('avg_pool').output, name= "ResNet50 without top layer") 
   
    elif sModelName == "EfficientNetB0":

        # load base model with top
        keBaseModel =  EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
        
		# Freeze the pretrained weights
        keBaseModel.trainable = False
		
        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(keBaseModel.output)
        x = layers.BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        x = layers.Dense(diFeature["nClasses"], activation="softmax", name="pred")(x)
        keModel = keras.models.Model(inputs=keBaseModel.input, outputs=x, name= "EfficientNetB0") 
		
        for layer in keModel.layers[-20:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        print(keModel.summary())
		
    elif sModelName == "lrcn":

        
        
        keModel = Sequential()

        keModel.add(Conv2D(16, 5, 5, input_shape=(256, 256, 3),  init= "he_normal",  activation='relu',  border_mode='same'))
        keModel.add(MaxPooling2D(pool_size=(2, 2)))
        keModel.add(Conv2D(20, 5, 5, init= "he_normal", activation='relu', border_mode='same'))
        keModel.add(MaxPooling2D(pool_size=(2, 2)))
        keModel.add(Conv2D(64, 3, 3, init= "he_normal",  activation='relu', border_mode='same'))
        keModel.add(MaxPooling2D(pool_size=(2, 2)))
        keModel.add(Conv2D(64, 3, 3, init= "he_normal",  activation='relu', border_mode='same'))
        keModel.add(MaxPooling2D(pool_size=(2, 2)))
        keModel.add(Conv2D(50, 3, 3, init= "he_normal", activation='relu', border_mode='same'))
        keModel.add(MaxPooling2D(pool_size=(2, 2)))
        
        keModel.add(Flatten())
		
    elif sModelName == "noModel":
		# we will not use this model but we created it to align with code structure
        
        
        keModel = Sequential()

        keModel.add(Conv1D(17, 1, input_shape=(25, 17),  activation='relu',  border_mode='same'))
        

    else: raise ValueError("Unknown 2D feature extraction model")

    # check input & output dimensions
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    print("Expected input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    if tuInputShape != diFeature["tuInputShape"]:
        raise ValueError("Unexpected input shape")
    if tuOutputShape != diFeature["tuOutputShape"]:
        print(str(diFeature["tuOutputShape"]))
        raise ValueError("Unexpected output shape")

    return keModel


def features_2D_predict_generator(sFrameBaseDir:str, sFeatureBaseDir:str, keModel:keras.Model, 
    nFramesNorm:int = 40, outputShape = None, sModelName='Other'):
    """
    Used by the MobileNet-LSTM NN architecture.
    The (video) frames (2-dimensional) in sFrameBaseDir are fed into keModel (eg MobileNet without top layers)
    and the resulting features are save to sFeatureBaseDir.
    """

    # do not (partially) overwrite existing feature directory
    #if os.path.exists(sFeatureBaseDir): 
    #    warnings.warn("\nFeature folder " + sFeatureBaseDir + " alredy exists, calculation stopped") 
    #    return

    # prepare frame generator - without shuffling!
    _, h, w, c = keModel.input_shape
    genFrames = FramesGenerator(sFrameBaseDir, 1, nFramesNorm, h, w, c, 
        liClassesFull = None, bShuffle=False)

    print("Predict features with %s ... " % keModel.name)
    nCount = 0
    # Predict - loop through all samples
    for _, seVideo in genFrames.dfVideos.iterrows():
        
        # ... sFrameBaseDir / class / videoname=frame-directory
        originalseVideo = seVideo.sFrameDir
        sVideoName = seVideo.sFrameDir.split("/")[-1]
        sLabel = seVideo.sLabel
        sFeaturePath = sFeatureBaseDir + "/" + sLabel + "/" + sVideoName + ".npy"

        # check if already predicted
        if os.path.exists(sFeaturePath):
            print("Video %5d: features already extracted to %s" % (nCount, sFeaturePath))
            nCount += 1
            continue
        print('-----------')
        print(originalseVideo)
        print(sVideoName)
        print('-----------')
        # get normalized frames and predict feature
        arX, _ = genFrames.data_generation(seVideo)
        if  sModelName == "noModel":
            arFeature = arX
        else:
            arFeature = keModel.predict(arX, verbose=0)

        # save to file
        os.makedirs(sFeatureBaseDir + "/" + sLabel, exist_ok = True)
        np.save(sFeaturePath, arFeature)

        print("Video %5d: features %s saved to %s" % (nCount, str(arFeature.shape), sFeaturePath))
        nCount += 1

    print("%d features saved to files in %s" % (nCount+1, sFeatureBaseDir))
    return
    
def features_2D_predict_generator_withResize(sFrameBaseDir:str, sFeatureBaseDir:str, keModel:keras.Model, 
    nFramesNorm:int = 40):
    """
    Used by the MobileNet-LSTM NN architecture.
    The (video) frames (2-dimensional) in sFrameBaseDir are fed into keModel (eg MobileNet without top layers)
    and the resulting features are save to sFeatureBaseDir.
    """

    # do not (partially) overwrite existing feature directory
    #if os.path.exists(sFeatureBaseDir): 
    #    warnings.warn("\nFeature folder " + sFeatureBaseDir + " alredy exists, calculation stopped") 
    #    return
	
	# Hamzah: Resize the images
	
    # prepare frame generator - without shuffling!
    _, h, w, c = keModel.input_shape
    genFrames = FramesGenerator(sFrameBaseDir, 1, nFramesNorm, h, w, c, 
        liClassesFull = None, bShuffle=False)

    print("Predict features with %s ... " % keModel.name)
    nCount = 0
    # Predict - loop through all samples
    for _, seVideo in genFrames.dfVideos.iterrows():
        
        # ... sFrameBaseDir / class / videoname=frame-directory
        sVideoName = seVideo.sFrameDir.split("/")[-1]
        sLabel = seVideo.sLabel
        sFeaturePath = sFeatureBaseDir + "/" + sLabel + "/" + sVideoName + ".npy"

        # check if already predicted
        if os.path.exists(sFeaturePath):
            print("Video %5d: features already extracted to %s" % (nCount, sFeaturePath))
            nCount += 1
            continue

        # get normalized frames and predict feature
        arX, _ = genFrames.data_generation(seVideo)
        arFeature = keModel.predict(arX, verbose=0)

        # save to file
        os.makedirs(sFeatureBaseDir + "/" + sLabel, exist_ok = True)
        np.save(sFeaturePath, arFeature)

        print("Video %5d: features %s saved to %s" % (nCount, str(arFeature.shape), sFeaturePath))
        nCount += 1

    print("%d features saved to files in %s" % (nCount+1, sFeatureBaseDir))
    return
	
	
def features_3D_predict_generator(sFrameBaseDir:str, sFeatureBaseDir:str, 
    keModel:keras.Model, nBatchSize:int = 16):
    """
    Used by I3D-top-only model.
    The videos (frames) are fed into keModel (=I3D without top layers) and
    resulting features are saved to disc. 
    (Later these features are used to train a small model containing 
    only the adjusted I3D top layers.)
    """

    # do not (partially) overwrite existing feature directory
    #if os.path.exists(sFeatureBaseDir): 
    #    warnings.warn("\nFeature folder " + sFeatureBaseDir + " alredy exists, calculation stopped") 
    #    return

    # prepare frame generator - without shuffling!
    _, nFramesModel, h, w, c = keModel.input_shape
    genFrames = FramesGenerator(sFrameBaseDir, nBatchSize, nFramesModel, h, w, c, 
        liClassesFull = None, bShuffle=False)

    # Predict
    print("Predict features with %s ... " % keModel.name)

    nCount = 0
    # loop through all samples
    for _, seVideo in genFrames.dfVideos.iterrows():

        # ... sFrameBaseDir / class / videoname=frame-directory
        sVideoName = seVideo.sFrameDir.split("/")[-1]
        sLabel = seVideo.sLabel
        sFeaturePath = sFeatureBaseDir + "/" + sLabel + "/" + sVideoName + ".npy"

        # check if already predicted
        if os.path.exists(sFeaturePath):
            print("Video %5d: features already extracted to %s" % (nCount, sFeaturePath))
            nCount += 1
            continue

        # get normalized frames
        arFrames, _ = genFrames.data_generation(seVideo)

        # predict single sample
        arFeature = keModel.predict(np.expand_dims(arFrames, axis=0))[0]

        # save to file
        os.makedirs(sFeatureBaseDir + "/" + sLabel, exist_ok = True)
        np.save(sFeaturePath, arFeature)

        print("Video %5d: features %s saved to %s" % (nCount, str(arFeature.shape), sFeaturePath))
        nCount += 1

    print("%d features saved to files in %s" % (nCount+1, sFeatureBaseDir))
    
    return