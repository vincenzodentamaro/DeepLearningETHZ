# Improving artist classification using data augmentation with GANs
This repository contains 6 programs all of which were used for our project:

-DCGAN: contains the program to generate new paintings according the DCGAN architecture based on https://github.com/carpedm20/DCGAN-tensorflow

-DCGAN_and_Resnet18: contains the program to classify the painting dataset augmented with GAN generated data based on https://github.com/Genoc/cs231n-final

-DAGAN-interpolation: contains the implementation of the DAGAN based on  https://github.com/AntreasAntoniou/DAGAN and an extension of the DAGAN architecture making interpolation between classes possible as was suggested in the feedback of our proposal

-BAGAN: contains the program to generate new paintings according the BAGAN architcture based on https://github.com/IBM/BAGAN. The BAGAN architecture is also used to generate new MNIST samples

-MNIST classifier : contains the program to classify different MNIST datasets constructed from real MNIST samples and GAN generated samples from the BAGAN to improve our understanding about using gan generated data for classification tasks.

## Readme files
All of the above mentioned directories have their own readme file for running the code.

## Download data
The used dataset, painters by numbers, can be downloaded from Kaggle: https://www.kaggle.com/c/painter-by-numbers/data
We used python scripts to extract a csv file (final_train_info.csv) with the paintings names by style (make_reduced_datafile.py). 
You can make a subfolder containing the selected files by running 'make_subselection_data.py'. However, you can also use the full dataset, as our models will only use the files mentioned in the csv file.




