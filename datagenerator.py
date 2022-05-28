"""
https://github.com/FrederikSchorr/sign-language

For neural network training the method Keras.model.fit_generator is used. 
This requires a generator that reads and yields training data to the Keras engine.
"""


import glob
import os
import sys
import cv2

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils

#from tensorflow.python.keras.utils.data_utils import Sequence


from frame import files2frames, images_normalize, frames_show, images_normalize_withGrayscale 

class FramesGenerator(tf.keras.utils.Sequence):
    """Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, sPath:str, \
        nBatchSize:int, nFrames:int, nHeight:int, nWidth:int, nChannels:int, \
        liClassesFull:list = None, bShuffle:bool = True, resizeImg:bool = False, convertToGrapy:bool = False):
        """
        Assume directory structure:
        ... / sPath / class / videoname / frames.jpg
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.nFrames = nFrames
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.nChannels = nChannels
        self.tuXshape = (nFrames, nHeight, nWidth, nChannels)
        self.bShuffle = bShuffle
        		
        # retrieve all videos = frame directories
        self.dfVideos = pd.DataFrame(sorted(glob.glob(sPath + "/*/*")), columns=["sFrameDir"])
        self.nSamples = len(self.dfVideos)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + sPath)
        print("Detected %d samples in %s ..." % (self.nSamples, sPath))

        # extract (text) labels from path
        seLabels =  self.dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        self.dfVideos.loc[:, "sLabel"] = seLabels
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfVideos.sLabel.unique()))

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfVideos.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfVideos.sLabel)
        
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # get batch of videos
        dfVideosBatch = self.dfVideos.loc[indexes, :]
        nBatchSize = len(dfVideosBatch)

        # initialize arrays
        arX = np.empty((nBatchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate data for single video(frames)
            #print(dfVideosBatch.iloc[i,:])
            arX[i,], arY[i] = self.__data_generation(dfVideosBatch.iloc[i,:])
        #print(arX.shape)

        # onehot the labels
        return arX, keras.utils.to_categorical(arY, num_classes=self.nClasses)


    def __data_generation(self, seVideo:pd.Series) -> (np.array(float), int):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        #ar_nFrames = files2frames(seVideo.sFrameDir, self.resizeImg, self.convertToGrapy)
        ar_nFrames = files2frames(seVideo.sFrameDir)

        # only use the first nChannels (typically 3, but maybe 2 for optical flow)
        ar_nFrames = ar_nFrames[..., 0:self.nChannels]
        
			
		
        ar_fFrames = images_normalize(ar_nFrames, self.nFrames, self.nHeight, self.nWidth, bRescale = True)
        
        return ar_fFrames, seVideo.nLabel

    def data_generation(self, seVideo:pd.Series) -> (np.array(float), int):
        return self.__data_generation(seVideo)

class FramesGenerator_withSplitting(tf.keras.utils.Sequence):
    """Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, sPath:str, \
        nBatchSize:int, nFrames:int, nHeight:int, nWidth:int, nChannels:int, \
        liClassesFull:list = None, bShuffle:bool = True, resizeImg:bool = False, convertToGrapy:bool = False):
        """
        Assume directory structure:
        ... / sPath / class / videoname / frames.jpg
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.nFrames = nFrames
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.nChannels = nChannels
        self.tuXshape = (nFrames, nHeight, nWidth, nChannels)
        self.bShuffle = bShuffle
        		
        # retrieve all videos = frame directories
        self.dfVideos = pd.DataFrame(sorted(glob.glob(sPath + "/*/*")), columns=["sFrameDir"])
        self.nSamples = len(self.dfVideos)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + sPath)
        print("Detected %d samples in %s ..." % (self.nSamples, sPath))

        # extract (text) labels from path
        seLabels =  self.dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        self.dfVideos.loc[:, "sLabel"] = seLabels
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfVideos.sLabel.unique()))

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfVideos.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfVideos.sLabel)
        
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # get batch of videos
        dfVideosBatch = self.dfVideos.loc[indexes, :]
        nBatchSize = len(dfVideosBatch)

        # initialize arrays
        arX = np.empty((nBatchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate data for single video(frames)
            arX[i,], arY[i] = self.__data_generation(dfVideosBatch.iloc[i,:])
            #print("Sample #%d" % (indexes[i]))

        # onehot the labels
        return arX, keras.utils.to_categorical(arY, num_classes=self.nClasses)


    def __data_generation(self, seVideo:pd.Series) -> (np.array(float), int):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        ar_nFrames = files2frames(seVideo.sFrameDir, self.convertToGrapy, self.resizeImg)

        # only use the first nChannels (typically 3, but maybe 2 for optical flow)
        ar_nFrames = ar_nFrames[..., 0:self.nChannels]
        
			
		
        ar_fFrames = images_normalize_withGrayscale(ar_nFrames, self.nFrames, self.nHeight, self.nWidth, bRescale = True)
        
        return ar_fFrames, seVideo.nLabel

    def data_generation(self, seVideo:pd.Series) -> (np.array(float), int):
        return self.__data_generation(seVideo)

class FeaturesGenerator_multi_withSplitting(tf.keras.utils.Sequence):
    """Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, sPath_01:str, sPath_02:str, nBatchSize:int, tuXshape_01, tuXshape_02, \
        liClassesFull:list = None, bShuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath_01 / class / videoname / frames.jpg
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.tuXshape_01 = tuXshape_01
        self.tuXshape_02 = tuXshape_02
        self.bShuffle = bShuffle
        		
        # retrieve all videos = frame directories
        self.dfSamples_01, self.nClasses_01, self.liClasses_01, self.nSamples_01 = self.initialize_gen( sPath_01, liClassesFull, tuXshape_01)
        self.dfSamples_02, self.nClasses_02, self.liClasses_02, self.nSamples_02 = self.initialize_gen( sPath_02, liClassesFull, tuXshape_02)
        
        self.on_epoch_end()
        return

    def initialize_gen(self, sPath, liClassesFull, tuXshape):
                #print(sPath )
        dfSamples = sPath
        #self.dfSamples = pd.DataFrame(sorted(glob.glob(sPath + "/*/*.npy")), columns=["sPath"])
        nSamples = len(dfSamples)
        if nSamples == 0: raise ValueError("Found no feature files in " + sPath)
        print("Detected %d samples in %s ..." % (nSamples, sPath))

        # test shape of first sample
        arX = np.load(dfSamples.sPath[0])
        if arX.shape != tuXshape: raise ValueError("Wrong feature shape: " + str(arX.shape) + str(tuXshape))

        # extract (text) labels from path
        seLabels =  dfSamples.sPath.apply(lambda s: s.split("/")[-2])
        dfSamples.loc[:, "sLabel"] = seLabels.astype(int) #hamzah: I added   .astype(int)       
        #print(self.dfSamples.loc[:, "sLabel"])		
           
        # extract unique classes from all detected labels
        liClasses = sorted(list(dfSamples.sLabel.unique()))
        #hamzah
        liClasses = [int(i) for i in liClasses]
        #print(self.liClasses)		

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull).astype(int))
            #print(liClassesFull)
            # check detected vs provided classes
            if set(liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            liClasses = liClassesFull
            
        nClasses = len(liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(liClasses)
        dfSamples.loc[:, "nLabel"] = trLabelEncoder.transform(dfSamples.sLabel)

        return dfSamples,  nClasses, liClasses, nSamples
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples_01 / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples_01)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # get batch of videos
        dfSamplesBatch_01 = self.dfSamples_01.loc[indexes, :]
        dfSamplesBatch_02 = self.dfSamples_02.loc[indexes, :]

        nBatchSize = len(dfSamplesBatch_01)

        # initialize arrays
        arX_01 = np.empty((nBatchSize, ) + self.tuXshape_01, dtype = float)
        arX_02 = np.empty((nBatchSize, ) + self.tuXshape_02, dtype = float)

        arY = np.empty((nBatchSize), dtype = int)
        arY_02 = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate data for single video(frames)
            arX_01[i,], arY[i] = self.__data_generation(dfSamplesBatch_01.iloc[i,:])
            arX_02[i,], arY_02[i] = self.__data_generation(dfSamplesBatch_02.iloc[i,:])
            #print("Sample #%d" % (indexes[i]))

        # onehot the labels
        arX = [arX_01, arX_02]
        return arX, keras.utils.np_utils.to_categorical(arY, num_classes=self.nClasses_01)


    def __data_generation(self, seSample:pd.Series) -> (np.array(float), int):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        'Generates data for 1 sample' 
        
        'Generates data for 1 sample' 
        try:
            arX = np.load(seSample.sPath)

            return arX, seSample.nLabel
        except:
            print(seSample.sPath)
            raise ValueError("Error "+seSample.sPath)

class FeaturesGenerator_multiInput(tf.keras.utils.Sequence):
    """return samples from two sources and check if they match
    """

    def __init__(self, sPath_01:str, sPath_02:str, nBatchSize:int, tuXshape_01, tuXshape_02, \
        liClassesFull:list = None, bShuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath_01 / class / videoname / frames.jpg
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.tuXshape_01 = tuXshape_01
        self.tuXshape_02 = tuXshape_02
        self.bShuffle = bShuffle
        self.sPath_02 = sPath_02
        		
        # retrieve all videos = frame directories
        self.dfSamples_01, self.nClasses_01, self.liClasses_01, self.nSamples_01 = self.initialize_gen(sPath_01, liClassesFull, tuXshape_01)
        ####self.dfSamples_02, self.nClasses_02, self.liClasses_02, self.nSamples_02 = self.initialize_gen( sPath_02, liClassesFull, tuXshape_02)
        
        self.on_epoch_end()
        return

    def initialize_gen(self, sPath, liClassesFull, tuXshape):
                #print(sPath )
        dfSamples = sPath
        #self.dfSamples = pd.DataFrame(sorted(glob.glob(sPath + "/*/*.npy")), columns=["sPath"])
        nSamples = len(dfSamples)
        if nSamples == 0: raise ValueError("Found no feature files in " + sPath)
        print("Detected %d samples in %s ..." % (nSamples, sPath))

        # test shape of first sample
        arX = np.load(dfSamples.sPath[0])
        if arX.shape != tuXshape: raise ValueError("Wrong feature shape: " + str(arX.shape) + str(tuXshape))

        # extract (text) labels from path
        seLabels =  dfSamples.sPath.apply(lambda s: s.split("/")[-2])
        dfSamples.loc[:, "sLabel"] = seLabels.astype(int) #hamzah: I added   .astype(int)       
        #print(self.dfSamples.loc[:, "sLabel"])		
           
        # extract unique classes from all detected labels
        liClasses = sorted(list(dfSamples.sLabel.unique()))
        #hamzah
        liClasses = [int(i) for i in liClasses]
        #print(liClasses)		

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull).astype(int))
            #print(liClassesFull)
            # check detected vs provided classes
            if set(liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            liClasses = liClassesFull
            
        nClasses = len(liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(liClasses)
        dfSamples.loc[:, "nLabel"] = trLabelEncoder.transform(dfSamples.sLabel)

        return dfSamples,  nClasses, liClasses, nSamples
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples_01 / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples_01)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # get batch of videos
        dfSamplesBatch_01 = self.dfSamples_01.loc[indexes, :]
        ###dfSamplesBatch_02 = self.dfSamples_02.loc[indexes, :]

        nBatchSize = len(dfSamplesBatch_01)

        # initialize arrays
        arX_01 = np.empty((nBatchSize, ) + self.tuXshape_01, dtype = float)
        arX_02 = np.empty((nBatchSize, ) + self.tuXshape_02, dtype = float)

        arY = np.empty((nBatchSize), dtype = int)
        arY_02 = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate data for single video(frames)

            # if fileName_01 != fileName_02:
            #     print("========Not matched=================")
            #     print(fileName_01)
            #     continue
            # elif ~os.path.exists(dfSamplesBatch_01.iloc[i,:].sPath):
            #     print("============= File not exist ================")
            #     print(dfSamplesBatch_01.iloc[i,:].sPath)
            #     continue
            # elif ~os.path.exists(dfSamplesBatch_02.iloc[i,:].sPath):
            #     print("============= File not exist ================")
            #     print(dfSamplesBatch_02.iloc[i,:].sPath)
            #     continue
           
            arX_01[i,], arY[i] = self.__data_generation(dfSamplesBatch_01.iloc[i,:])

            fileName_01 = os.path.basename(dfSamplesBatch_01.iloc[i,:].sPath)
            signID = dfSamplesBatch_01.iloc[i,:].sPath.split('/')
            fileName_02 = os.path.join(self.sPath_02, signID[-2], fileName_01.replace('npy','jpg'))
            # print("self.sPath_02: " + self.sPath_02)
            # print("input 01: " + fileName_01)
            # print("signID: " + str(signID))
            # print(fileName_02)
            # print("*******************************************")
            # print(arX_02.shape)
            arX_02[i,], arY_02[i] = self.__data_generation_v2(fileName_02, dfSamplesBatch_01.iloc[i,:].nLabel)
            #print("Sample #%d" % (indexes[i]))

        # onehot the labels
        arX = [arX_01, arX_02]
        return arX, keras.utils.np_utils.to_categorical(arY, num_classes=self.nClasses_01)


    def __data_generation(self, seSample:pd.Series) -> (np.array(float), int):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        'Generates data for 1 sample' 
        
        'Generates data for 1 sample' 
        try:
            arX = np.load(seSample.sPath)

            return arX, seSample.nLabel
        except:
            print(seSample.sPath)
            raise ValueError("Error "+seSample.sPath)

    def __data_generation_v2(self, filePath, nlabel) -> (np.array(float), int):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        'Generates data for 1 sample' 
        
        'Generates data for 1 sample' 
        try:
            #arX = np.load(filePath)
            arX = cv2.imread(filePath)
            arX = cv2.resize(arX, self.tuXshape_02[0:2])
            
            return arX, nlabel
        except Exception as e:
            #print(filePath)
            print(e)
            print(self.tuXshape_02[0:2])
            raise ValueError("Error "+filePath)


class FeaturesGenerator(tf.keras.utils.Sequence):
    """Reads and yields (preprocessed) I3D features for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, sPath:str, nBatchSize:int, tuXshape, \
        liClassesFull:list = None, bShuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath / class / feature.npy
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.tuXshape = tuXshape
        self.bShuffle = bShuffle

        # retrieve all feature files
        self.dfSamples = pd.DataFrame(sorted(glob.glob(sPath + "/*/*.npy")), columns=["sPath"])
        self.nSamples = len(self.dfSamples)
        if self.nSamples == 0: raise ValueError("Found no feature files in " + sPath)
        print("Detected %d samples in %s ..." % (self.nSamples, sPath))

        # test shape of first sample
        arX = np.load(self.dfSamples.sPath[0])
        if arX.shape != tuXshape: raise ValueError("Wrong feature shape: " + str(arX.shape) + str(tuXshape))

        # extract (text) labels from path
        seLabels =  self.dfSamples.sPath.apply(lambda s: s.split("/")[-2])
        self.dfSamples.loc[:, "sLabel"] = seLabels.astype(int) #hamzah: I added   .astype(int)       
        #print(self.dfSamples.loc[:, "sLabel"])		
           
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfSamples.sLabel.unique()))
        #hamzah
        self.liClasses = [int(i) for i in self.liClasses]
        #print(self.liClasses)		

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull).astype(int))
            #print(liClassesFull)
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfSamples.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfSamples.sLabel)
        print('in Initalization');
        
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # Find selected samples
        dfSamplesBatch = self.dfSamples.loc[indexes, :]
        nBatchSize = len(dfSamplesBatch)

        # initialize arrays
        arX = np.empty((nBatchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate single sample data
            arX[i,], arY[i] = self.__data_generation(dfSamplesBatch.iloc[i,:])

        # onehot the labels
        return arX, keras.utils.to_categorical(arY, num_classes=self.nClasses)

    def __data_generation(self, seSample:pd.Series) -> (np.array(float), int):
        'Generates data for 1 sample' 

        arX = np.load(seSample.sPath)

        return arX, seSample.nLabel

class FeaturesGenerator_withSplitting(tf.keras.utils.Sequence):
    """Reads and yields (preprocessed) I3D features for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, sPath:str, nBatchSize:int, tuXshape, \
        liClassesFull:list = None, bShuffle:bool = True, diVideoSet=None, diFeature = None):
        """
        Assume directory structure:
        ... / sPath / class / feature.npy
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.tuXshape = tuXshape
        self.diFeature = diFeature
        self.bShuffle = bShuffle
        self.diVideoSet = diVideoSet
        self.newInputShape = (self.diVideoSet["nFramesNorm"], self.diFeature["tuInputShape"][0],self.diFeature["tuInputShape"][1],self.diFeature["tuInputShape"][2] )
        # retrieve all feature files
        #print(sPath )
        self.dfSamples = sPath
        #self.dfSamples = pd.DataFrame(sorted(glob.glob(sPath + "/*/*.npy")), columns=["sPath"])
        self.nSamples = len(self.dfSamples)
        if self.nSamples == 0: raise ValueError("Found no feature files in " + sPath)
        print("Detected %d samples in %s ..." % (self.nSamples, sPath))

        # test shape of first sample
        arX = np.load(self.dfSamples.sPath[0])
        if self.diVideoSet != None and self.diVideoSet["reshape_input"] == True: 
            arX = arX.reshape(self.newInputShape)
        else:
            if arX.shape != tuXshape: raise ValueError("Wrong feature shape: " + str(arX.shape) + str(tuXshape))

        # extract (text) labels from path
        seLabels =  self.dfSamples.sPath.apply(lambda s: s.split("/")[-2])
        self.dfSamples.loc[:, "sLabel"] = seLabels.astype(int) #hamzah: I added   .astype(int)       
        #print(self.dfSamples.loc[:, "sLabel"])		
           
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfSamples.sLabel.unique()))
        #hamzah
        self.liClasses = [int(i) for i in self.liClasses]
        #print(self.liClasses)		

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull).astype(int))
            #print(liClassesFull)
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfSamples.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfSamples.sLabel)
        
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # Find selected samples
        dfSamplesBatch = self.dfSamples.loc[indexes, :]
        nBatchSize = len(dfSamplesBatch)

        # initialize arrays
        arX = np.empty((nBatchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate single sample data
            arX[i,], arY[i] = self.__data_generation(dfSamplesBatch.iloc[i,:])
        #print(arX.shape)
        # onehot the labels
        return arX, tf.keras.utils.to_categorical(arY, num_classes=self.nClasses)

    def __data_generation(self, seSample:pd.Series) -> (np.array(float), int):
        'Generates data for 1 sample' 
        
        'Generates data for 1 sample' 
        try:
            #print(seSample.sPath)
            arX = np.load(seSample.sPath)
            #print(arX.shape)
            if self.diVideoSet != None and self.diVideoSet["reshape_input"] == True: 
                arX = arX.reshape(self.newInputShape) 

            return arX, seSample.nLabel
        except:
            print(seSample.sPath)
            raise ValueError("Error "+seSample.sPath)


class VideoClasses():
    """
    Loads the video classes (incl descriptions) from a csv file
    """
    def __init__(self, sClassFile:str):
        # load label description: index, sClass, sLong, sCat, sDetail
        self.dfClass = pd.read_csv(sClassFile)

        # sort the classes
        self.dfClass = self.dfClass.sort_values("sClass").reset_index(drop=True)
        
        self.liClasses = list(self.dfClass.sClass)
        self.nClasses = len(self.dfClass)

        print("Loaded %d classes from %s" % (self.nClasses, sClassFile))
        return