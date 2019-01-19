# BAGAN
Keras implementation of [Balancing GAN (BAGAN)](https://arxiv.org/abs/1803.09655). This implementation was largely taken from https://github.com/IBM/BAGAN. See below for our contributions.

# Packages:

To run it on the cluster, make sure to load python_gpu/3.6.6 and hdf5


# To run BAGAN on painting dataset:
```
python bagan_train_paint.py -u 1 -e 150
```
IMPORTANT: 
-The training painting dataset should be located a directory "/train_reduced/" 

-The testing painting dataset should be located in a directory "/test_reduced"

-Furthermore both of these datasets should be furnished with a csv file, respectively "/final_reduced_train_info.csv" and "/final_reduced_test_info.csv". These csv files must contain at least two columns "new_filename", denoting the file names of all the paintings in our dataset and "style", denoting the style aka the class of the corresponding painting. These files can be constructed from the data csv file in ../data_info_files/final_train_info_modified.csv.

-The actual implementation is one based on 3 classes: Romanticism, Realism and Impressionism. But can be easily accomodated to more classes, by making some changes in the file "./rw/batch_generator_paint.py"

-As stated in the report, this GAN gives bad results when it's run on the painting dataset. This GAN was used to generate MNIST samples, see next section.

The trained architectures with some example pictures can afterwards be found in a result folder.


# To run BAGAN on mnist dataset:
```
python bagan_train_mnist.py -u 1 -a 1000 -e 5
```
This will train the BAGAN on the first 1000 MNIST samples, but the "a" or amount flag can be changed according to the task .

Thus by using a=1000, the GAN will be trained on the dataset refered to as in the report MNIST1000_GAN1000_same. By using a=50000, 
the GAN will be trained on the dataset refered to as in the report MNIST1000_GAN1000_full. 

The trained architectures with some example pictures can afterwards be found in a result folder.

In order to generate samples from these in the form of numpy arrays, you have to rename these result folders in a characteristic name, for examples MNIST1000_GAN1000_full or MNIST1000_GAN1000_same and run the next bit of code.

# To generate samples from one of these GAN's use:
```
python generate_samples.py -f "name of the result folder containing the weights" -e 0
```
A 1000 GAN samples will be generated in the same folder with both pictures and numpy arrays. These samples in the form of numpy arrays
can be later used to augment the dataset. In the repository ../MNIST-classifier, we use these numpy arrays on an MNIST classifier (In this repository we already give these numpy arrays to run the code more easily)


# Contributions to the code:

1. Created the file rw/batch_generator_paint.py to apply the BAGAN on a painting dataset, generalizing the MNIST file
2. Created the file generate_samples.py to easily generate samples from a certain input GAN architecture
3. Various other small adjustments to the code were made.
