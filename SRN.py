# -*- coding: utf-8 -*-


from dataclasses import asdict
import os
import glob
import time
import sys
import warnings
import seaborn as sn
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import   EarlyStopping,  TimeHistory
from tensorflow.keras.optimizers import Adam as adam

import keras

import tensorflow as tf


from datagenerator import VideoClasses, FeaturesGenerator_multiInput
from model_lstm import lstm_build_multi_single

 
def readData(dataPath, convertToInt = False, ext= ".npy"):
    # The csv file should have the following columns (available in dataset folder)
    data_all = pd.read_csv(dataPath,names=['index','cat','sPath','framesN','signerID'],header=None)
    if convertToInt:
        dfSamples = data_all.sPath.to_frame()
        dfSamples.sPath = dfSamples.sPath + ext
        seLabels = pd.get_dummies(data_all.sPath.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()
    else:
        dfSamples = data_all['sPath'].copy().to_frame()
        dfSamples.sPath = dfSamples.sPath + ext
        seLabels = data_all.sPath.apply(lambda s: s.split("/")[-2])
 
    return dfSamples, seLabels




def train_feature_generator(sFeatureDir:str, sModelDir:str, sLogPath:str, keModel:keras.Model, oClasses: VideoClasses,
    nBatchSize:int=16, nEpoch:int=100, fLearn:float=1e-4, expFullPath=None, val_available= True, csvFile = False, trainPath_01 = None, valPath_01=None, testPath_01=None, trainPath_02 = None, valPath_02=None, testPath_02=None, loadModel = False, savedModel=None): 
    if csvFile:
        dfSamples, seLabels  = readData(trainPath_01)

        if testPath_01 == None:
            dfSamples, dfSamples_test_01, y_train, y_val = train_test_split( dfSamples, seLabels, test_size=0.10,  stratify=seLabels)
            dfSamples.reset_index(drop=True, inplace=True)
            seLabels  = dfSamples.sPath.apply(lambda s: s.split("/")[-2])

            dfSamples_test_01.reset_index(drop=True, inplace=True)
            seLabels_test_01 =  pd.get_dummies(dfSamples_test_01.sPath.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()
        else:
            dfSamples_test_01, seLabels_test_01 = readData(testPath_01, True)


        indices = np.arange(dfSamples.shape[0])
        train_data_01, val_data_01, _, _, _, _  = train_test_split(dfSamples, seLabels, indices, test_size=0.20, stratify=seLabels)			
        train_data_01.reset_index(drop=True, inplace=True)
        val_data_01.reset_index(drop=True, inplace=True)
        
        
        print(train_data_01.shape)
        
        
    else:  
        # This does not support fusion yet
        dfSamples = pd.DataFrame(sorted(glob.glob(sFeatureDir+ "/train" + "/*/*.npy")), columns=["sPath"])
        seLabels =  dfSamples.sPath.apply(lambda s: s.split("/")[-2])
        
        train_data, val_data, _, _ = train_test_split( dfSamples, seLabels, test_size=0.20, random_state=42, stratify=seLabels)
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        dfSamples_test = pd.DataFrame(sorted(glob.glob(sFeatureDir+ "/test" + "/*/*.npy")), columns=["sPath"])
        seLabels_test =  pd.get_dummies(dfSamples_test.sPath.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()
        
    
        
    if loadModel == False:
        genFeaturesTrain = FeaturesGenerator_multiInput(train_data_01, trainPath_02, nBatchSize,  keModel.input_shape[0][1:], keModel.input_shape[1][1:], oClasses.liClasses, bShuffle=True)
        genFeaturesVal = FeaturesGenerator_multiInput(val_data_01, trainPath_02, nBatchSize,  keModel.input_shape[0][1:], keModel.input_shape[1][1:], oClasses.liClasses, bShuffle=True)
        genFeaturesTest   = FeaturesGenerator_multiInput(dfSamples_test_01, testPath_02, nBatchSize,  keModel.input_shape[0][1:], keModel.input_shape[1][1:], oClasses.liClasses, bShuffle=False)

    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger(sLogPath.split(".")[0] + "-acc.csv")

	# Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(monitor='val_loss', patience=10)

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointBest = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/model-best.h5",
        verbose = 1, save_best_only = True)

    optimizer = adam(lr = fLearn)
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    time_callback = TimeHistory()
    
    print("Fit with generator, learning rate %f ..." % fLearn)
    from math import ceil
    n_points = seLabels_test_01.shape[0]
    steps_per_epoch = ceil(n_points / nBatchSize)

    if loadModel:
        keModel = []
        keModel = tf.keras.models.load_model(savedModel)
        genFeaturesTest   = FeaturesGenerator_multiInput(dfSamples_test_01, testPath_02, nBatchSize,  keModel.input_shape[0][1:], keModel.input_shape[1][1:], oClasses.liClasses, bShuffle=False)
        es = 0
    else:
        hist = keModel.fit(
            genFeaturesTrain,
            validation_data = genFeaturesVal,
            epochs = nEpoch, #nEpoch
            workers = 1, #1,                 
            use_multiprocessing = False, #False,
            verbose = 1, 
            steps_per_epoch=steps_per_epoch,
            callbacks=[csv_logger, time_callback, checkpointBest, early_stopper]) 
        
        genFeaturesTrain =[]
        genFeaturesVal = []

        es = early_stopper.get_epochNumber() 
        timeInfo = time_callback.times 
        saveModelTimes(timeInfo, expFullPath)
        writeResults(hist, expFullPath,   0)   
        visualizeHis(hist, expFullPath, 0)
  
    test(keModel, es, expFullPath,False,  oClasses, genFeaturesTest, None, seLabels_test_01) 

    return


def saveModelTimes(timeInfo, expPath):
    # Print and save model times
    print('Total time:')
    print(np.sum(timeInfo))
    writeCSVFile(timeInfo , 'model_times.csv', expPath)

def writeCSVFile(data, fileName, pathName):
    pd.DataFrame(data).to_csv(os.path.join(pathName,fileName), sep=',')


def writeResults(hist, expPath, useBatch=0):
    
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    

    dataFrameData = np.transpose([train_loss, train_acc, val_loss, val_acc])
    
    writeCSVFile(dataFrameData , 'train_val_losses_acc.csv', expPath)


def test(rm, es, mainExpFolder, load_to_memory, class_limit, test_generator=None, X_test=None, y_test=None):
    
    if load_to_memory:
        #use X_test, y_test
        loss, acc = rm.evaluate(X_test, y_test, verbose=0)
    else:
        loss, acc = rm.evaluate(test_generator, verbose=0)
        #loss, acc = rm.evaluate(X_test, y_test, verbose=0)
    
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
    
    f= open(mainExpFolder+ '/testAccurracy.txt',"a")

    earlyStoppingEpoch = es
    f.write('Early stopped at:\t' +str(earlyStoppingEpoch) + '\t'+ str(loss) + ' '+ str(acc)+ '\n')
   
    if load_to_memory: 
        Y_pred = rm.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
        printConfusionMatrix(X_test, y_test, class_limit, y_pred, mainExpFolder)

    else:
        #Y_pred = rm.predict(X_test)
        Y_pred = rm.predict_generator(test_generator, 
        workers = 1,                 
        use_multiprocessing = False,
        verbose = 1)
	
        y_pred = Y_pred.argmax(axis=1)

        printConfusionMatrix(X_test, y_test, class_limit, y_pred, mainExpFolder)

def printConfusionMatrix(X_test, y_test, numb_classes, y_pred, mainExpFolder):
    #
    confusionMatrix = confusion_matrix(y_test.argmax(axis=1), y_pred)
    print('Confusion Matrix: ', confusionMatrix.shape)
    filename =os.path.join(mainExpFolder,'CM.csv')
     
    pd.DataFrame(confusionMatrix).to_csv(filename, sep=',')
    pd.DataFrame(y_test.argmax(axis=1)).to_csv(os.path.join(mainExpFolder ,'testLabels.csv'), sep=',')
    pd.DataFrame(y_pred).to_csv(os.path.join(mainExpFolder ,'predicted_testLabels.csv'), sep=',')
       
    # Visualizing of confusion matrix
    plotCM = False
    if plotCM:
        df_cm = pd.DataFrame(confusionMatrix, range(numb_classes), range(numb_classes))
        plt.figure(figsize = (numb_classes,numb_classes))
        sn.set(font_scale=1.4)#for label size
        sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
        plt.savefig(os.path.join(mainExpFolder,'cm.png'))
    
    classMetrics = classification_report(y_test.argmax(axis=1), y_pred, output_dict=True)
    print(classification_report(y_test.argmax(axis=1), y_pred))
    df = pd.DataFrame(classMetrics).transpose()
    pd.DataFrame(df).to_csv(os.path.join(mainExpFolder, 'ResultMetrics.csv'), sep=',')

def visualizeHis(hist, experName, useBatch):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    
    xc=range(len(val_acc))

    plt.figure(3)
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model loss')
    plt.grid(True)
    plt.legend(['Train','Validation'])
    plt.savefig(experName +'/loss.png')
    if useBatch == 1:
        plt.show()

    plt.figure(4)
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.grid(True)
    plt.legend(['Train','Validation'],loc=4)

    plt.savefig(experName +'/acc.png')
    if useBatch==1:
        plt.show()
    
def train_mobile_lstm(diVideoSet, diFeature_01, diFeature_02,  \
                          expFullPath = None, expPath = None, val_available = None, \
			csvFile = None, trainPath_01 = None, valPath_01 = None, testPath_01 = None,\
        trainPath_02 = None, valPath_02 = None, testPath_02 = None, \
            sClassFile = None,   \
			sImageFeatureDir = None,  loadModel = False, savedModel=None):

    if os.path.exists(expFullPath)  == False:
        os.mkdir(expFullPath,0o755)


    


    sModelDir = expFullPath 

    print("\nStarting training ...")
    print(os.getcwd())

    # read the classes
    oClasses = VideoClasses(sClassFile)

    #Load and train the model
    sLogPath = os.path.join(expFullPath, time.strftime("%Y%m%d-%H%M", time.gmtime()) + "-%s%03d-image-mobile-lstm.csv"%(diVideoSet["sName"], diVideoSet["nClasses"]))
    print("Image log: %s" % sLogPath)

    keModelImage = lstm_build_multi_single(diVideoSet["nFramesNorm"], diFeature_01["tuOutputShape"][0], diFeature_02["tuOutputShape"][0], oClasses.nClasses, fDropout = 0.5)
    train_feature_generator(sImageFeatureDir, sModelDir, sLogPath, keModelImage, oClasses,
        nBatchSize = 32, nEpoch = 1000, fLearn = 1e-4, expFullPath=expFullPath, val_available=val_available,
        csvFile=csvFile,trainPath_01=trainPath_01,  valPath_01=valPath_01, testPath_01=testPath_01, trainPath_02=trainPath_02,  valPath_02=valPath_02, testPath_02=testPath_02, loadModel = loadModel, savedModel=savedModel)


    return
    
    
if __name__ == '__main__':


    diVideoSet = {"sName" : "muSign",
    "nClasses" : 64,   # number of classes
    "nFramesNorm" : 18,    # number of frames per video
    "nMinDim" : 224,   # smaller dimension of saved video-frames
    "tuShape" : (224, 224), # height, width
    "nFpsAvg" : 30,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 2.0} # seconds 
    
    # feature extractor 

    diFeature_01 = {"sName" : "mobilenet", # select the pre-trained model 
        "tuInputShape" : (224, 224, 3),
        "tuOutputShape" : (1024, )}

    #diFeature_02 = {"sName" : "vgg16",
    #    "tuInputShape" : (224, 224, 3),
    #    "nOutput" : 512,
    #    "tuOutputShape" : (512, )}

    diFeature_02 = {"sName" : "mobilenet", # select the pre-trained model 
        "tuInputShape" : (224, 224, 3),
        "tuOutputShape" : (1024, )}
    
    
    sClassFile_all       = "./data/ISA_64.csv" #path to clase names: karsl_502.csv, karsl_190.csv, ISA_64.csv
    sImageFeatureDir_all = ['']*3

    
    dataSetHomePath ='./dataset/ISA64/'	

    trainPath_01_all = [dataSetHomePath+'test_01.csv', dataSetHomePath+'test_01.csv', dataSetHomePath+'test_01.csv']  

    
    valPath_01_all = [''] * 3
    testPath_01_all = [ dataSetHomePath+'001.csv', dataSetHomePath+'001.csv', dataSetHomePath+'001.csv']  



    # # The second data source is for image/single file datas (single file/image for the whole video). Just path since path of first source will be be replaced by new path
    dataSetHomePath_02 ='/home/eye/lsa64_raw/fusion/forward/'	
    dataSetHomePath_b = '/home/eye/lsa64_raw/fusion/backward/'
    dataSetHomePath_both = '/home/eye/lsa64_raw/fusion/both/'

    trainPath_02_all = [
     dataSetHomePath_02+'01', dataSetHomePath_b+'01', dataSetHomePath_both+'01']

    valPath_02_all = [''] * 3
    testPath_02_all = [
    dataSetHomePath_02+'01', dataSetHomePath_b+'01', dataSetHomePath_both+'01']
    
    expPath_all = ['lsa64_401_v4', 'lsa64_402_v4', 'lsa64_403_v4', 'lsa64_501_v4', 'lsa64_502_v4', 'lsa64_503_v4', 'lsa64_601_v4', 'lsa64_602_v4', 'lsa64_603_v4'] 




    loadModel_all = [False] * 3
    savedModel_all=[None] * 3

    csvFile_all = [True] * 3

    val_available_all =[False] * 3
    i = 0
    print('==========================================')
    print(len(expPath_all))
    while i< len(expPath_all):
        print(expPath_all[i])
        expPath = expPath_all[i]
        val_available = val_available_all[i]
        expFullPath = os.path.join(os.getcwd(),'results',expPath)
        
        
        train_mobile_lstm(diVideoSet, diFeature_01, diFeature_02,  \
                          expFullPath=expFullPath, expPath=expPath, val_available=val_available, \
			csvFile = csvFile_all[i],         trainPath_01 = trainPath_01_all[i], valPath_01 = valPath_01_all[i], testPath_01 = testPath_01_all[i],\
        trainPath_02 = trainPath_02_all[i], valPath_02 = valPath_02_all[i], testPath_02 = testPath_02_all[i], \
            sClassFile = sClassFile_all,  
			sImageFeatureDir = sImageFeatureDir_all[i],  loadModel = loadModel_all[i], savedModel=savedModel_all[i])


        
        i = i + 1