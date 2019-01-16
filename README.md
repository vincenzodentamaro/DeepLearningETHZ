# Improving artist classification using data augmentation with GANs
This files explains how to run the various used models in the GAN data augmentation.

## Download data
The used dataset, painters by numbers, can be downloaded from Kaggle: https://www.kaggle.com/c/painter-by-numbers/data
We used python scripts to extract a csv file (final_train_info.csv) with the paintings names by style (make_reduced_datafile.py). 
You can make a subfolder containing the selected files by running 'make_subselection_data.py'. However, you can also use the full dataset, as our models will only use the files mentioned in the csv file.

## Data augmentation GAN (DAGAN)

### Data
Make sure that the data is situated in the folder './DeepLearningETHZ/train_reduced/

### Dependencies
tqdm 4.11.2
tensorflow-gpu 1.6.0
tensorboard 1.6.0
scipy 1.1.0
scikit-image 0.14.0
pillow 4.3.0
numpy 1.14.1
imageio 2.4.1

### Run experiments on leonhard cluster
all experiments were run on the leonhard cluster with the python_gpu/3.6.4 module and imagio installed
Move to the folder DAGAN-interpolation, where all the files are situated
For generating figures 3.2a and 3.2b (Resnet structure of generator) run the following command
```
bsub -n 20 -W 120:00 -R "rusage[mem=4500, ngpus_excl_p=1]" python train_painting_styles.py --batch_size 8 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title paintingstylesResNet2 --num_of_gpus 1 --z_dim 300 --dropout_rate_value 0.5 --infofile ../data_info_files/final_train_info.csv --data_dir ../train_reduced
```
For genreating figures 3.2c and 3.2d (UResnet structure of generator) first uncomment line 297 and 343 in dagan_architectures.py and then run the same command as above.
For generating the interpolations on the omniglot dataset, run the following command
```
bsub -n 20 -W 24:00 -R "rusage[mem=4500, ngpus_excl_p=1]" python train_omniglot_dagan.py --batch_size 32 --generator_inner_layers 3 --discriminator_inner_layers 5 --num_generations 64 --experiment_title omniglot --num_of_gpus 1 --z_dim 100 --dropout_rate_value 0.5
```
Plots of the losses and gradients are saved to the logs folder in the experimant folder (which is automatically generated). These plots can be checked in tensorboard.

### Extra information on the files
For the implementation of the DAGAN, we started from the github implementation of Antreas Antoniou https://github.com/AntreasAntoniou/DAGAN
First we made an interface between our painting dataset and their implementation of the DAGAN. This newly made interface is located in data.py (compare with their implementation data_old.py to see that we made a substantial effort to make an easy-to-work-with interface)

