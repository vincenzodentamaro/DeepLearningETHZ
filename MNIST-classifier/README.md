# MNIST-classifier
This file contains the necessary code to run our experiment on MNIST, in which we tried to grasp why GAN generated data may not also be 
useful for classification. 

The GAN-generated_DATA folder contains a 1000 samples for each class in the form of numpy arrays from our 3 differently trained GANs.

# To run the classifier

python classifier_train.py --a "amount MNIST samples used for the classifier, the default is 1000" --augmentation "0, 1, 2 or 3"


for 
