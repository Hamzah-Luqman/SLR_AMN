The repository contains the code used in "Hamzah Luqman, **'An  Efficient Two-Stream Network for Sign Language Recognition Using Accumulative Video Motion'. IEEE Access (2022)"** paper.

## Requirments
This project has been developed using Tensorflow 2.5 and Keras 2.5. 

## Dataset
Two datasets have been utilized in this paper:

- <a href="https://github.com/Hamzah-Luqman/KArSL"> KArSL </a> 
- <a href="http://facundoq.github.io/datasets/lsa64/"> ISA64 </a>

After extracting the features of each dataset, I depended on csvfiles as an input to the generator (except for AMN which can read them directly from folders). The csv file should have the following format: (category, Sign/Class, fullPath, framesN, SignerID). The most important columns are Sign/Class and fullPath. Other columns can contain any values. Samples of these csv files for ISA64 dataset are available in the Dataset folder. 

## How to run the code?
### _prepareData.py_ 
This code converts the sign videos into frames/images. It also extracts the features from each sign and saves them into a specific folder.

### _AVM.m_
Generate the AVM images from the sign frames.

### _DMN.py_
This is the dynamic motion network stream.

### _AMN.py_
This is the accumulative motion network stream. It accepts AVM images extracted using AVM.m.

### _SRN.py_
This is the sign recognition network stream. This network accepts two inputs: sign features and its AVM images.


