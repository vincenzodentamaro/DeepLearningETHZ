# MNIST-classifier
This file contains the necessary code to run our experiment on MNIST, in which we tried to grasp why GAN generated data may not also be 
useful for classification. 

The GAN-generated_DATA folder contains a 1000 samples for each class in the form of numpy arrays from our 3 differently trained GANs.

# To run the classifier

python classifier_train.py --a "amount MNIST samples used for the classifier, the default is 1000" --augmentation "0, 1, 2 or 3"

use the flag "--augmentation 0" for no GAN generated samples extra
use the flag "--augmentation 1" for 1000 GAN generated samples from the MNIST1000_GAN1000_same GAN 
use the flag "--augmentation 2" for 1000 GAN generated samples from the MNIST1000_GAN1000_random GAN 
use the flag "--augmentation 3" for 1000 GAN generated samples from the MNIST1000_GAN1000_full GAN 

# To run the loop_classifier:
./loop_classifier

This will save the test accuracy to a csv file "results.csv" over 5 different runs.
