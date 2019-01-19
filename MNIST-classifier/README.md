# MNIST-classifier
This file contains the necessary code to run our classification task on MNIST, in which we tried to grasp why GAN generated data may not be useful for classification. 

The MNIST classifier was taken from the Deep Learning Exercise 5. Our contribution was adapting this code to feed different MNIST construced datasets (MNIST1000_GAN1000_same,MNIST1000_GAN1000_random,MNIST1000_GAN1000_full, look at the report for the differences between these datasets) into the classifier and various other little changes.

The GAN-generated_DATA folder already contains 200 samples for each class in the form of numpy arrays from our 3 differently trained GANs:

-1000_class_i.npy: contains 200 samples of the number "i" from the MNIST1000_GAN1000_same GAN

-samples_class_i.npy: contains 200 samples of the number "i" from the MNIST1000_GAN1000_random GAN

-50000samples_class_i.npy: contains 200 samples of the number "i" from the MNIST1000_GAN1000_full GAN


For how these samples can be obtained, take a look at the ../BAGAN folder.

# Packages:

To run it on the cluster, make sure to load python_gpu/3.6.6 and hdf5

# To run the classifier
```
python classifier_train.py --a "amount MNIST samples used for the classifier, the default is 1000" --augmentation "0, 1, 2 or 3"
```

use the flag "--augmentation 0" for no GAN generated samples extra
use the flag "--augmentation 1" for 1000 GAN generated samples from the MNIST1000_GAN1000_same GAN 
use the flag "--augmentation 2" for 1000 GAN generated samples from the MNIST1000_GAN1000_random GAN 
use the flag "--augmentation 3" for 1000 GAN generated samples from the MNIST1000_GAN1000_full GAN 

# To run the loop_classifier:
```
./loop_classifier
```
This will save the test accuracy to a csv file "results.csv" over 5 different runs.
