The repository contains the code used in "Hamzah Luqman, **'An  Efficient Two-Stream Network for Sign Language Recognition Using Accumulative Video Motion'. IEEE Access (2022)"** paper.

## Requirments
This project has been developed using Tensorflow 2.5 and Keras 2.5. 

## Dataset
Two datasets have been utilized in this paper:

- <a href="https://github.com/Hamzah-Luqman/KArSL"> KArSL </a> 
- <a href="http://facundoq.github.io/datasets/lsa64/"> ISA64 </a>

After extacting the features of each dataset, I depended on a CSV files as an input to the generator (except for AMN that can read them directly from folders). The csv file should the following format: (category, Sign/Class, fullPath, framesN, SignerID). The most important columns are Sign/Class and fullPath. Other columns can contain any values. Samples of these csv file for ISA64 dataset are available in Dataset folder. 
