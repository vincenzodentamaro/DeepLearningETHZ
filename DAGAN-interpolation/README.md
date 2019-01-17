# Data augmentation GAN (DAGAN)

## Data
Make sure that the data is situated in the folder './DeepLearningETHZ/train_reduced/

## Dependencies
tqdm 4.11.2
tensorflow-gpu 1.6.0
tensorboard 1.6.0
scipy 1.1.0
scikit-image 0.14.0
pillow 4.3.0
numpy 1.14.1
imageio 2.4.1

## Run experiments on leonhard cluster
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

## Extra information on the files
1. The experiments are run by first calling train_painting_styles.py. This file generates a dataset object (object class defined in data.py), creates an experiment object and runs the experiment.
2. The experiment object is defined in experiment_builder.py. When creating the experiment, the graph of the DAGAN is initialized by making the DAGAN object. When running the experiment, the graph is executed repeatedly on the input batches.
3. The DAGAN object is defined in dagan_networks_wgan.py, which makes use of the generator and discriminator objects defined in dagan_architectures.py
4. Various util functions are situated in the utils folder

## Contributions to the code
For the implementation of the DAGAN, we started from the github implementation of Antreas Antoniou https://github.com/AntreasAntoniou/DAGAN
1. First we made an interface between our painting dataset and their implementation of the DAGAN. This newly made interface is located in data.py (compare with their implementation data_old.py to see that we made a substantial effort to make an easy-to-work-with interface)
2. Then we added a consistency term in the loss method of the DAGAN object (dagan_networks_wgan.py)
3. We also added various extra summaries to visualize the training
4. For the interpolation, we added the following methods to the DAGAN object: encode, interpolate_inter_class and interpolate_intra_class. We added various helpfunctions, such as create_interpolation_interval (which uses spherical sampling) in utils/interpolations.py and interpolation_generator in utils/sampling to visualize the results. 
5. Various other small adjustments to the code were made.