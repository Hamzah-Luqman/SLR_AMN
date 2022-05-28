# -*- coding: utf-8 -*-
"""
Accept two data sources and fuse them at the classification layer
@author: hluqman@kfupm.edu.sa
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam as adam
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from tensorflow.keras import layers
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.utils import shuffle 


from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import   EarlyStopping,  TimeHistory, TestCallback_gen
#  


def writeResults(hist, expPath,  useBatch=0):
    
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['accuracy']
    val_acc=hist.history['val_accuracy']
    

    dataFrameData = np.transpose([train_loss, train_acc, val_loss, val_acc])
    
    writeCSVFile(dataFrameData , 'train_val_losses_acc.csv', expPath)


def saveModelTimes(timeInfo, expPath):
    # Print and save model times
    print('Total time:')
    print(np.sum(timeInfo))
    writeCSVFile(timeInfo , 'model_times.csv', expPath)

def writeCSVFile(data, fileName, pathName):
    pd.DataFrame(data).to_csv(os.path.join(pathName,fileName), sep=',')

def visualizeHis(hist, experName, useBatch=0):
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
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    #plt.style.use(['classic'])

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

def printConfusionMatrix(y_test, numb_classes, y_pred, mainExpFolder):
    #
    confusionMatrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: ', confusionMatrix.shape)
    filename =os.path.join(mainExpFolder,'CM.csv')
     
    pd.DataFrame(confusionMatrix).to_csv(filename, sep=',')
    pd.DataFrame(y_test).to_csv(os.path.join(mainExpFolder ,'TestLabels.csv'), sep=',')
    pd.DataFrame(y_pred).to_csv(os.path.join(mainExpFolder ,'predicted_testLabels.csv'), sep=',')
       
    # Visualizing of confusion matrix
    plotCM = False
    if plotCM:
        df_cm = pd.DataFrame(confusionMatrix, range(numb_classes), range(numb_classes))
        plt.figure(figsize = (numb_classes,numb_classes))
        sn.set(font_scale=1.4)#for label size
        sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
        plt.savefig(os.path.join(mainExpFolder,'cm.png'))
    
    classMetrics = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    df = pd.DataFrame(classMetrics).transpose()
    pd.DataFrame(df).to_csv(os.path.join(mainExpFolder, 'ResultMetrics.csv'), sep=',')


def pretrainedModel(img_size, modelName, retrainModel=False, dataFormat='folder'):

    input_img = tf.keras.layers.Input(shape=(img_size, img_size, 3))
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

    cnn_out = keras.layers.Dense(class_limit, activation="softmax")(cnn_out)
    model = tf.keras.models.Model(model_cnn.input, cnn_out)
    if dataFormat =='csv':
        model.compile(metrics=['accuracy'], loss="categorical_crossentropy",optimizer=adam(learning_rate=1e-4))
    else:
        model.compile(metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=adam(learning_rate=1e-4))

    return model

def append_image_extention(imgPath):
    return imgPath+'.jpg'

def main(modelName, retrainModel, train_path, test_path, mainExpFolder, model_dest, class_limit, img_size, dataFormat='folder', loadModel = False, savedModel=None):
    if os.path.exists(mainExpFolder)  == False :
        os.mkdir(mainExpFolder, 0o755)

    print('---------------------------------------START----------------------------')
    print(mainExpFolder)
    print('------------------------------------------------------------------------')
    datagen_train = ImageDataGenerator(validation_split=0.10, rotation_range=10, zoom_range=0.2) #for ISA64 dataset
    #datagen_train = ImageDataGenerator(validation_split=0.20) #for KArSL
    datagen_test = ImageDataGenerator()

    if dataFormat == 'csv':
        train_df = pd.read_csv(train_path,names=['index','label','sPath','framesN','signerID'],header=None, dtype=str)
        train_df = shuffle(train_df)
        train_df["sPath"] = train_df["sPath"].apply(append_image_extention)

        if test_path != None:
            test_df = pd.read_csv(test_path, names=['index','label','sPath','framesN','signerID'],header=None, dtype=str)
            test_df["sPath"] = test_df.sPath.apply(append_image_extention)
            print(test_df)
            train_ds = datagen_train.flow_from_dataframe(dataframe=train_df, directory=None, x_col="sPath", y_col="label", batch_size=64, class_mode='categorical', subset = "training", target_size=(img_size, img_size), shuffle=True, seed=42)
            val_ds  = datagen_train.flow_from_dataframe(dataframe=train_df,  directory=None, x_col="sPath", y_col="label", batch_size=64, class_mode='categorical', subset = "validation", target_size=(img_size, img_size), shuffle=True, seed=42)
            test_ds = datagen_test.flow_from_dataframe(dataframe=test_df,  directory=None, x_col="sPath", y_col="label", batch_size=64, class_mode='categorical', target_size=(img_size, img_size), shuffle=False)

        else:
            y_ = train_df['label'].to_numpy()
            data_pdf = train_df[['sPath', 'label']]
            X_train_df, test_df = train_test_split(data_pdf ,  test_size=0.20, stratify=y_ )

            print(data_pdf)
            print(X_train_df)
            datagen_train = ImageDataGenerator(validation_split=0.10, rotation_range=10, zoom_range=0.2) #for ISA64 dataset
            #datagen_train = ImageDataGenerator(validation_split=0.20) #for KArSL

            train_ds = datagen_train.flow_from_dataframe(dataframe=X_train_df, directory=None, x_col="sPath", y_col="label", batch_size=64, class_mode='categorical', subset = "training", target_size=(img_size, img_size), shuffle=True, seed=42)
            val_ds  = datagen_train.flow_from_dataframe(dataframe=X_train_df,  directory=None, x_col="sPath", y_col="label", batch_size=64, class_mode='categorical', subset = "validation", target_size=(img_size, img_size), shuffle=True, seed=42)
            test_ds = datagen_test.flow_from_dataframe(dataframe=test_df,  directory=None, x_col="sPath", y_col="label", batch_size=64, class_mode='categorical', target_size=(img_size, img_size), shuffle=False)
        print('///////////////////////')
        print(train_ds.n)
         
        STEP_SIZE_TRAIN= train_ds.n // 64
        STEP_SIZE_VALID= val_ds.n // 64

    else:


        train_ds = datagen_train.flow_from_directory(train_path,  batch_size=64, class_mode='sparse', target_size=(img_size, img_size), shuffle=True, seed=42, subset = "training")
        val_ds = datagen_train.flow_from_directory(train_path,  batch_size=64, class_mode='sparse', target_size=(img_size, img_size), shuffle=True, seed=42, subset = "validation")
        test_ds = datagen_test.flow_from_directory(test_path,  batch_size=64, class_mode='sparse', target_size=(img_size, img_size), shuffle=False)

        STEP_SIZE_TRAIN= train_ds.samples // 64
        STEP_SIZE_VALID= val_ds.samples // 64


    # model 
    if modelName =='mobileNet' and loadModel == False:
        model = pretrainedModel(img_size, modelName, retrainModel, dataFormat)

    early_stopper = EarlyStopping(monitor='val_loss', patience=10)
    time_callback = TimeHistory()
    checkpointBest = keras.callbacks.ModelCheckpoint( filepath = mainExpFolder + "/model-best.h5",  verbose = 1, save_best_only = True)

    if loadModel:
        model = keras.models.load_model(savedModel)
        print(model.summary())
        es = 0
    else:    
        hist = model.fit(train_ds, validation_data=val_ds, epochs = 1000, callbacks=[time_callback, checkpointBest, early_stopper])
        #save last model
        model.save(model_dest) 
        model = keras.models.load_model(filepath = mainExpFolder + "/model-best.h5")
        es = early_stopper.get_epochNumber() 
        timeInfo = time_callback.times 
        saveModelTimes(timeInfo, mainExpFolder)
        writeResults(hist, mainExpFolder, 0)     
        visualizeHis(hist, mainExpFolder, 0)

        # test
        score, acc = model.evaluate(test_ds)
        print('Test score:', score)
        print('Test accuracy:', acc)


    Y_pred = model.predict(test_ds)
    y_pred = np.argmax(Y_pred, axis=1)
    pd.DataFrame(y_pred).to_csv(os.path.join(mainExpFolder ,'predicted_testLabels.csv'), sep=',')


    ### Get labels
    y_test = test_ds.labels 
    pd.DataFrame(y_test).to_csv(os.path.join(mainExpFolder ,'testLabels.csv'), sep=',')

    ###
    Y_pred = model.predict(test_ds)
    y_pred = np.argmax(Y_pred, axis=1)
    pd.DataFrame(y_pred).to_csv(os.path.join(mainExpFolder ,'predicted_testLabels.csv'), sep=',')

    printConfusionMatrix(y_test, class_limit, y_pred, mainExpFolder)


### Model configurations ######
modelName = 'mobileNet'
input_size = 224
retrainModel = False
dataFormat = 'folder' # 'csv' for csv files or 'folder' to read from folders

#################

train_path = "/home/eye/lsa64_raw/fusion/forward/test_01" #contains train signs after forward fusion
test_path  = "/home/eye/lsa64_raw/fusion/forward/001" #contains test signs after forward fusion
mainExpFolder = './results/IAS64_412/'
model_dest = os.path.join(mainExpFolder, 'Last.h5')
class_limit = 64
loadModel = False
savedModel='./results/IAS64_412/model-best.h5'
main(modelName, retrainModel, train_path, test_path, mainExpFolder, model_dest, class_limit, input_size, dataFormat, loadModel, savedModel)

