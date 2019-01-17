# BAGAN
Keras implementation of [Balancing GAN (BAGAN)](https://arxiv.org/abs/1803.09655). This implementation was largely taken from https://github.com/IBM/BAGAN. Specific changes were applied to accomodate our specific tasks.

# To run BAGAN on painting dataset:

python bagan_train_paint.py -u 1 -e 150

IMPORTANT: the training painting dataset should be located a directory "/train_reduced/"
the testing painting dataset should be located in a directory "/test_reduced"
Furthermore both of these datasets should be furnished with a csv file, respectively "/final_reduced_train_info.csv" and "/final_reduced_test_info.csv"
These csv files must contain at least two columns "new_filename", denoting the file names of all the paintings in our dataset and "style", denoting the style aka the class of the corresponding painting.
The actual implementation is one based on 3 classes: Romanticism, Realism and Impressionism. But can be easily accomodated to more classes, by making some changes in the file "./rw/batch_generator_paint.py"


The trained architectures with some example pictures can afterwards be found in a result folder.


# To run BAGAN on mnist dataset:

python bagan_train_mnist.py -u 1 -a 1000 -e 5

This will train the BAGAN on the first 1000 MNIST samples, but the "a" or amount flag can be changed according to the task .

The trained architectures with some example pictures can afterwards be found in a result folder.

The MNIST1000_GAN1000_full, MNIST1000_GAN1000_random, MNIST1000_GAN1000_same folders contain these architectures with samples for the three GAN's that were considered in our report. 

# To generate samples from one of these GAN's use:

python generate_samples.py -f MNIST1000_GAN1000_full -e 0

the samples will be generated in the same folder as both pictures and numpy arrays.

To test these samples on a classifier, 1000 gan generated samples for each of these 3 GAN's are saved in the GAN-generated_DATA	folder.
