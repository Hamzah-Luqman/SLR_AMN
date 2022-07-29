"""
	https://github.com/FrederikSchorr/sign-language

Train a LSTM neural network to classify videos. 
Requires as input pre-computed features, 
calculated for each video (frames) with MobileNet.
"""

import os
import glob
import time
import sys
import warnings
import seaborn as sn
from matplotlib import pyplot as plt

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import   EarlyStopping,  TimeHistory, TestCallback_gen

import keras

from datagenerator import VideoClasses, FeaturesGenerator, FeaturesGenerator_withSplitting
from model_lstm import lstm_build


def train_feature_generator(sFeatureDir:str, sModelDir:str, sLogPath:str, keModel:keras.Model, oClasses: VideoClasses,
    nBatchSize:int=16, nEpoch:int=100, fLearn:float=1e-4, expFullPath=None, val_available= True, csvFile = False, trainPath = None, valPath=None, testPath=None, diVideoSet=None, diFeature = None, loadModel = False, savedModel=None) :
    print(diVideoSet    )
    if csvFile:
        print('===============================================')
        train_data = pd.read_csv(trainPath,names=['index','cat','sPath','framesN','signerID'],header=None)
        #print(train_data_all)
        #train_data_all = train_data_all.sample(frac=1)
        train_data = train_data['sPath'].copy().to_frame()
        train_data.sPath = train_data.sPath + ".npy"
        #print(dfSamples.sPath)
        seLabels =  train_data.sPath.apply(lambda s: s.split("/")[-2]) 

        if testPath == None:
            train_data, dfSamples_test, y_train, y_val = train_test_split( train_data, seLabels, test_size=0.10, random_state=42, stratify=seLabels)
            train_data.reset_index(drop=True, inplace=True)
            seLabels  = train_data.sPath.apply(lambda s: s.split("/")[-2])

            dfSamples_test.reset_index(drop=True, inplace=True)
            seLabels_test =  pd.get_dummies(dfSamples_test.sPath.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()
            #print(train_data)
            
            #dfSamples.sPath = train_data.sPath
        else:

            test_data = pd.read_csv(testPath,names=['index','cat','sPath','framesN','signerID'],header=None)
            dfSamples_test = test_data.sPath.to_frame()
            dfSamples_test.sPath = dfSamples_test.sPath + ".npy"
            seLabels_test =  pd.get_dummies(test_data.sPath.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()

        # vladiation part
        train_data, val_data, y_train, y_val = train_test_split( train_data, seLabels, test_size=0.20, random_state=42, stratify=seLabels)
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        
        
    else:            
        dfSamples = pd.DataFrame(sorted(glob.glob(sFeatureDir+ "/train" + "/*/*.npy")), columns=["sPath"])
        seLabels =  dfSamples.sPath.apply(lambda s: s.split("/")[-2])
        
        train_data, val_data, y_train, y_val = train_test_split( dfSamples, seLabels, test_size=0.30, random_state=42, stratify=seLabels)
        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)

        dfSamples_test = pd.DataFrame(sorted(glob.glob(sFeatureDir+ "/test" + "/*/*.npy")), columns=["sPath"])
        seLabels_test =  pd.get_dummies(test_data.sPath.apply(lambda s: s.split("/")[-2]).to_numpy().astype(int)).to_numpy()

    print(keModel.input_shape)
    genFeaturesTrain = FeaturesGenerator_withSplitting(train_data, nBatchSize,  keModel.input_shape[1:], oClasses.liClasses, True, diVideoSet = diVideoSet, diFeature = diFeature)
    genFeaturesVal = FeaturesGenerator_withSplitting(val_data, nBatchSize,  keModel.input_shape[1:], oClasses.liClasses, True, diVideoSet = diVideoSet, diFeature = diFeature)
    genFeaturesTest   = FeaturesGenerator_withSplitting(dfSamples_test, nBatchSize, keModel.input_shape[1:], oClasses.liClasses, False, diVideoSet = diVideoSet, diFeature = diFeature)
    
	
        
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger(sLogPath.split(".")[0] + "-acc.csv")

	# Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(monitor='val_loss', patience=20)

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointLast = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-last.h5",
        verbose = 0)
    checkpointBest = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/model-best.h5",
        verbose = 1, save_best_only = True)

    optimizer = keras.optimizers.Adam(lr = 1e-4) #1e-4
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    time_callback = TimeHistory()
    
    # Fit!
    print("Fit with generator, learning rate %f ..." % fLearn)
    print('******************************')
    print(loadModel)
    print('******************************')

    if loadModel:
        keModel = tf.keras.models.load_model(savedModel)
        es = 0
    else:    
        hist = keModel.fit(
        genFeaturesTrain,
            validation_data = genFeaturesVal,
            epochs = nEpoch,
            workers = 1, #4,                 
            use_multiprocessing = False, #True,
            verbose = 1, 
            callbacks=[csv_logger, time_callback,  checkpointBest, early_stopper]) 
        
        #Y_pred = keModel.predict_generator(genFeaturesTest)
        #y_pred = np.argmax(Y_pred, axis=1)
        keModel = tf.keras.models.load_model(sModelDir + "/model-best.h5")
        timeInfo = time_callback.times 
        saveModelTimes(timeInfo, expFullPath)
        es = early_stopper.get_epochNumber() 
        writeResults(hist, expFullPath,   0)     
        visualizeHis(hist, expFullPath, 0)


    test_loss, test_acc = keModel.evaluate(genFeaturesTest, verbose=0)
    print('\nTesting loss: {}, acc: {}\n'.format(test_loss, test_acc))
 
    test(keModel, es, expFullPath,False,  50, genFeaturesTest, None, seLabels_test) 


 
    
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
        loss, acc = rm.evaluate_generator(test_generator, verbose=0)
        #loss, acc = rm.evaluate(X_test, y_test, verbose=0)
    
    print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
    
    f= open(mainExpFolder+ '/testAccurracy.txt',"a")
    #f.write(' '.join(expInfo)+'\n')
    earlyStoppingEpoch = es
    f.write('Early stopped at:\t' +str(earlyStoppingEpoch) + '\t'+ str(loss) + ' '+ str(acc)+ '\n')
   
    if load_to_memory: 
        Y_pred = rm.predict(X_test)
        y_pred = np.argmax(Y_pred, axis=1)
        printConfusionMatrix(X_test, y_test, class_limit, y_pred, mainExpFolder)

    else:
        #Y_pred = rm.predict(X_test)
        Y_pred = rm.predict_generator(test_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        #y_pred = (y_pred > 0.5)
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
#    if useBatch==1:
#        plt.show()
    
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
    plt.legend(['Train','Validation','Test'],loc=4)

    plt.savefig(experName +'/acc.png')
    if useBatch==1:
        plt.show()
    
def train_model_lstm(diVideoSet, diFeature, expFullPath=None, expPath=None, val_available=None,
			sFolder = None,sClassFile = None, sVideoDir = None, sImageDir = None, sImageFeatureDir = None, sOflowDir = None, 
			sOflowFeatureDir = None, csvFile = False, trainPath = None, valPath = None, testPath = None, loadModel = False, savedModel=None):

    if os.path.exists(expFullPath)  == False:
        os.mkdir(expFullPath,0o755)


   

    sModelDir = expFullPath #"model"

    print("\nStarting training ...")
    print(os.getcwd())

    # read the classes
    oClasses = VideoClasses(sClassFile)

    # Image: Load and train the model
        
    sLogPath = os.path.join(expFullPath, time.strftime("%Y%m%d-%H%M", time.gmtime()) + "-%s%03d-image-mobile-lstm.csv"%(diVideoSet["sName"], diVideoSet["nClasses"]))
    print("Image log: %s" % sLogPath)
    keModelImage = lstm_build(diVideoSet["nFramesNorm"], diFeature["tuOutputShape"][0], oClasses.nClasses, fDropout = 0.5, modelName = diFeature["sName"])
    train_feature_generator(sImageFeatureDir, sModelDir, sLogPath, keModelImage, oClasses,
        nBatchSize = 32, nEpoch = 1000, fLearn = 1e-4, expFullPath=expFullPath, val_available=val_available,
        csvFile=csvFile,trainPath=trainPath,  valPath=valPath, testPath=testPath, diVideoSet=diVideoSet, diFeature = diFeature, loadModel = loadModel, savedModel=savedModel)

    return
    
    
if __name__ == '__main__':


    diVideoSet = {"sName" : "KArSL", 
    "nClasses" : 64,   # number of classes
    "nFramesNorm" : 18,    # number of frames per video
    "nMinDim" : 224,   # smaller dimension of saved video-frames
    "tuShape" : (224, 224), # height, width
    "nFpsAvg" : 30,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 2.0,# seconds 
    "Transformer":False,
    "reshape_input": False}  #True: if the raw input is different from the requested shape for the model
	
	
    # feature extractor 
    # diFeature = {"sName" : "mobilenet",
    #     "tuInputShape" : (224, 224, 3),
    #     "tuOutputShape" : (1024, )} # was 1024

    # diFeature = {"sName" : "mobilenetV3",
    #    "tuInputShape" : (224, 224, 3),
    #    "tuOutputShape" : (1280, )} # was 1280


    #diFeature = {"sName" : "InceptionResNetV2",
    #    "tuInputShape" : (256, 256, 3),
    #    "tuOutputShape" : (1024, )}
    #diFeature = {"sName" : "mobilenet_scratch",
    #    "tuInputShape" : (256, 256, 3),
    #    "tuOutputShape" : (256, 256, 3)}
#    diFeature = {"sName" : "xception_musign",
#        "tuInputShape" : (299, 299, 3),
#        "nOutput" : 2048,
#        "tuOutputShape" : (2048, )}

    #diFeature = {"sName" : "xception_musign",
    #    "tuInputShape" : (299, 299, 3),
    #    "nOutput" : 2048,
    #    "tuOutputShape" : (2048, )}
	
    diFeature = {"sName" : "Xception",
    "tuInputShape" : (299, 299, 3),
    "nOutput" : 2048,
    "tuOutputShape" : (2048, )}
	
    #diFeature = {"sName" : "lrcn",
    #"tuInputShape" : (256, 256, 3),
    #"nOutput" : 3200,
    #"tuOutputShape" : (3200, )}
    # diFeature = {"sName" : "EfficientNetB0",
    # "tuInputShape" : (224, 224, 3),
    # "nOutput" : 1280,
    # "tuOutputShape" : (1280, ),
    # "nClasses" : 190}

    # diFeature = {"sName" : "vgg16",
    #     "tuInputShape" : (224, 224, 3),
    #     "nOutput" : 512,
    #     "tuOutputShape" : (512, )}

    # diFeature = {"sName" : "ResNet50",
    #    "tuInputShape" : (224, 224, 3),
    #    "nOutput" : 2048,
    #    "tuOutputShape" : (2048, )}

    # diFeature = {"sName" : "ResNet50",
    #    "tuInputShape" : (224, 224, 3),
    #    "nOutput" : 512,
    #    "tuOutputShape" : (512, )}
      
    #diFeature = {"sName" : "lstm",
    #    "tuInputShape" : (128*128),
    #    "nOutput" : 128*128,
    #    "tuOutputShape" : (128*128, )}
    #diFeature = {"sName" : "lstm",
    #    "tuInputShape" : (17),
    #    "nOutput" : 17,
    #    "tuOutputShape" : (17, )}
    # dataSetHomePath ='dataset/lsa64_raw/features/18/mobilenet/color/'	
    #dataSetHomePath ='dataset/KArSLFrames/features/18/mobilenet_v3/'	
    #dataSetHomePath ='dataset/KArSLFrames/features/18/vgg16/'	
    dataSetHomePath ='/dataset/features/18/xception/'
    #dataSetHomePath ='dataset/features/18/ResNet50_2/ResNet50/'	
    sFolder_all = [None, None, None, None, None, None, None, None, None, None]*5
    sClassFile_all       = "./data/ISA_64.csv"
    sVideoDir_all        = [None, None, None, None, None, None, None, None, None, None]*5
    sImageDir_all        = [None, None, None, None, None, None, None, None, None, None]*5
    sImageFeatureDir_all = [None, None, None, None, None, None, None, None, None, None]*5
    sOflowDir_all        = [None, None, None, None, None, None, None, None, None, None]*5 
    sOflowFeatureDir_all = [None, None, None, None, None, None, None, None, None, None]*5 

    csvFile_all = [True]*21
    loadModel_all = [False]*21
    savedModel_all=[None]*21
       
	

    trainPath_all = [dataSetHomePath+'/test_01.csv',
	    dataSetHomePath+'/test_02.csv',
		dataSetHomePath+'/test_03.csv',
		dataSetHomePath+'/test_04.csv',dataSetHomePath+'/test_05.csv',
		dataSetHomePath+'/test_06.csv', dataSetHomePath+'/test_07.csv', dataSetHomePath+'/test_08.csv', dataSetHomePath+'/test_09.csv', dataSetHomePath+'/test_10.csv']
    valPath_all = ['','','','','','','','','','','']*5

    testPath_all = [dataSetHomePath+'/all.csv',dataSetHomePath+'/001.csv',
	    dataSetHomePath+'/002.csv',
		dataSetHomePath+'/003.csv',
		dataSetHomePath+'/004.csv',dataSetHomePath+'/005.csv',
		dataSetHomePath+'/006.csv', dataSetHomePath+'/007.csv', dataSetHomePath+'/008.csv', dataSetHomePath+'/009.csv', dataSetHomePath+'/010.csv']
    expPath_all = ['IAS64_color_0011','IAS64_color_0012','IAS64_color_0013','IAS64_color_0014','IAS64_color_0015','IAS64_color_0016','IAS64_color_0017', 'IAS64_color_0018', 'IAS64_color_0019', 'IAS64_color_0020', 'IAS64_color_0021']



    val_available_all =[False]*22

    i = 0
    while i< len(expPath_all):
	
        print(expPath_all[i])
        expPath = expPath_all[i]
        val_available = val_available_all[i]
        expFullPath = os.path.join(os.getcwd(),'results',expPath)
        
        train_model_lstm(diVideoSet, diFeature, expFullPath=expFullPath, expPath=expPath, val_available=val_available,
			sFolder = sFolder_all[i],sClassFile = sClassFile_all, sVideoDir = sVideoDir_all[i], sImageDir = sImageDir_all[i],
			sImageFeatureDir = sImageFeatureDir_all[i], sOflowDir = sOflowDir_all[i], sOflowFeatureDir = sOflowFeatureDir_all[i],
			csvFile = csvFile_all[i], trainPath = trainPath_all[i], valPath = valPath_all[i], testPath = testPath_all[i], loadModel = loadModel_all[i], savedModel=savedModel_all[i])
        
        i = i + 1
