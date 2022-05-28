"""
https://github.com/FrederikSchorr/sign-language

Define or load a Keras LSTM model.
"""

from multiprocessing.dummy import active_children
import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from  tensorflow.keras.layers import Bidirectional, LSTM, TimeDistributed, BatchNormalization, Conv3D, MaxPooling1D
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.optimizers import Adam as adam
from tensorflow.keras.regularizers import l2

#from tensorflow import keras
#from tensorflow.python.keras.layers import Dense, Activation, Flatten, Dropout, ZeroPadding3D, Input, concatenate
#from tensorflow.keras.models import Model

from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import ELU, ReLU, LeakyReLU

###########import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, ZeroPadding3D, Input, concatenate, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras import layers

from transformer_model import TransformerEncoder, PositionalEmbedding

def lstm_build(nFramesNorm:int, nFeatureLength:int, nClasses:int, fDropout:float = 0.5, modelName:str = 'None') : #-> tensorflow.keras.Model

    # Build new LSTM model
    print("Build and compile LSTM model ...")
    
    print('Model Name: ', modelName)			

    if modelName == 'lstm':
        keModel = tf.keras.models.Sequential()
        keModel.add(keras.layers.LSTM(nFeatureLength * 1, return_sequences=False, dropout=fDropout))
        keModel.add(keras.layers.Dense(nClasses, activation='softmax'))
    

    else:
        
        
        keModel = tf.keras.models.Sequential()
        keModel.add(tf.keras.layers.LSTM(2024, return_sequences=True,
            input_shape=(nFramesNorm, nFeatureLength),
            dropout=0.7))
        keModel.add(tf.keras.layers.LSTM(2024, return_sequences=False, dropout=0.7))
        
        keModel.add(tf.keras.layers.Flatten())
        keModel.add(tf.keras.layers.Dense(1024, activation='relu'))
        keModel.add(tf.keras.layers.Dropout(0.6))
        keModel.add(tf.keras.layers.Dropout(0.6))

        keModel.add(tf.keras.layers.Dense(nClasses, activation='softmax'))

    print("================MODEL SUMMARY==================")
    keModel.summary()
    print("================END OF MODEL SUMMARY==================")

    return keModel

def lstm_build_multi(nFramesNorm:int, nFeatureLength_01:int, nFeatureLength_02:int, nClasses:int, fDropout:float = 0.5, modelName:str = 'None'): #-> keras.Model:

    # Build new LSTM model
    print("Build and compile the model ...")

    
    input_01 = Input(shape=(nFramesNorm, nFeatureLength_01))         
    model_01_01 = tf.keras.layers.LSTM(nFeatureLength_01 * 1, return_sequences=True, input_shape=(nFramesNorm, nFeatureLength_01), dropout=fDropout)(input_01)    
    model_01_02 = tf.keras.layers.LSTM(nFeatureLength_01 * 1, return_sequences=False, dropout=fDropout)(model_01_01)


    input_02 = Input(shape=(nFramesNorm, nFeatureLength_02))         
    model_02_01 = tf.keras.layers.LSTM(1024, return_sequences=True, input_shape=(nFramesNorm, nFeatureLength_02), dropout=0.5)(input_02)    
    model_02_02 = tf.keras.layers.LSTM(1024, return_sequences=False, dropout=0.5)(model_02_01)

    mergedLayers = concatenate([model_01_02, model_02_02])         
    fc = Dense(nClasses, activation='softmax')(mergedLayers)
    
    keModel = Model([input_01, input_02], [fc]) 
       
    
    keModel.summary()

    return keModel

def lstm_build_multi_single(nFramesNorm:int, nFeatureLength_01:int, nFeatureLength_02:int, nClasses:int, fDropout:float = 0.5, modelName:str = 'None'): #-> keras.Model:

    # Build a fused LSTM and CNN (LSTM for frames and CNN for a single image per video)
 

    ## Model 01
    input_frames =  Input(shape=(nFramesNorm, nFeatureLength_01), name='input_frames')
    
    x1 = LSTM(2048, return_sequences=True, input_shape=(nFramesNorm, nFeatureLength_01), dropout=0.7)(input_frames)    
    x1 = LSTM(2048, return_sequences=True, dropout=0.7)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dropout(0.6)(x1)
    x1 = tf.keras.layers.Dropout(0.6)(x1)

    #########x1 = BatchNormalization( axis = -1 )(x1)


    ## Model 02
    img_size = 224
    input_img = tf.keras.layers.Input(shape=(img_size, img_size, 3), name='input_img')
    x2 = preprocess_input(input_img)
    x2 = tf.keras.applications.MobileNet(weights="imagenet",include_top=False, input_tensor=x2)
    x2 = keras.layers.GlobalAveragePooling2D()(x2.output)
    x2 = keras.layers.Dropout(0.6)(x2)
    x2 = keras.layers.Dropout(0.6)(x2)

    #########x2 = BatchNormalization(axis = -1 )(x2)

    ##
    x = concatenate([x1, x2])
    x = BatchNormalization(axis = -1 )(x)
    x = x[:,:,np.newaxis]
    x = tf.keras.layers.Conv1D(256, 7, activation='relu')(x)
    # x = keras.layers.Dropout(0.6)(x)
    # x = keras.layers.Dropout(0.6)(x)

    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dropout(0.7)(x)
    x = keras.layers.Dropout(0.6)(x)
    x = Flatten()(x)

    fc = layers.Dense(nClasses,  activation = "softmax")(x)
    #keModel = tf.keras.models.Model([input_frames, model_cnn.input], fc)
    keModel = tf.keras.models.Model(inputs=[input_frames, input_img], outputs=fc)

    
    keModel.summary()

    return keModel


def pretrainedModel(img_size, modelName, nClasses, retrainModel=False):

    input_img = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    #normalization_layer = tf.keras.layers.Rescaling(1./255)
    if modelName == 'mobileNet':
        input_img = preprocess_input(input_img)
        model_cnn = tf.keras.applications.MobileNet(weights="imagenet",include_top=False, input_tensor=input_img)
    elif modelName == 'InceptionV3':
        model_cnn = tf.keras.applications.InceptionV3(weights="imagenet",include_top=False, input_shape=(256, 256, 3))
    if retrainModel:
        for layer in model_cnn.layers[:-4]:
            layer.trainable = True

    cnn_out = keras.layers.GlobalAveragePooling2D()(model_cnn.output)
    cnn_out = keras.layers.Dropout(0.6)(cnn_out)
    cnn_out = keras.layers.Dropout(0.6)(cnn_out)

    #cnn_out = keras.layers.Dense(nClasses, activation="softmax")(cnn_out)
    model = tf.keras.models.Model(model_cnn.input, cnn_out)
    #model.compile(metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=adam(learning_rate=1e-4))
    
    return model


def lstm_load(sPath:str, nFramesNorm:int, nFeatureLength:int, nClasses:int) -> tf.keras.Model:

    print("Load trained LSTM model from %s ..." % sPath)
    keModel = tf.keras.models.load_model(sPath)
    
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    print("Loaded input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    if tuInputShape != (nFramesNorm, nFeatureLength):
        raise ValueError("Unexpected LSTM input shape")
    if tuOutputShape != (nClasses, ):
        raise ValueError("Unexpected LSTM output shape")

    return keModel