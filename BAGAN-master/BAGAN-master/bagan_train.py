"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

from collections import defaultdict
from PIL import Image

import numpy as np

from optparse import OptionParser
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class BatchGenerator:

    TRAIN = 1
    TEST = 0

    def __init__(self, data_src, batch_size=32, class_to_prune=None, unbalance=0):
        self.batch_size = batch_size
        self.data_src = data_src

        # Load data
        mnist = input_data.read_data_sets("dataset/mnist", one_hot=False)

        assert self.batch_size > 0, 'Batch size has to be a positive integer!'

        if self.data_src == self.TEST:
            self.dataset_x = mnist.test.images
            self.dataset_y = mnist.test.labels
        else:
            self.dataset_x = mnist.train.images
            self.dataset_y = mnist.train.labels

        # Normalize between -1 and 1
        self.dataset_x = (np.reshape(self.dataset_x, (self.dataset_x.shape[0], 28, 28)) - 0.5) * 2

        # Include 1 single color channel
        self.dataset_x = np.expand_dims(self.dataset_x, axis=1)

        assert (self.dataset_x.shape[0] == self.dataset_y.shape[0])

        # Compute per class instance count.
        classes = np.unique(self.dataset_y)
        self.classes = classes
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))

        # Prune if needed!
        if class_to_prune is not None:
            all_ids = list(np.arange(len(self.dataset_x)))

            mask = [class_to_prune == lc for lc in self.dataset_y]
            all_ids_c = np.array(all_ids)[mask]
            np.random.shuffle(all_ids_c)

            other_class_count = np.array(per_class_count)
            other_class_count = np.delete(other_class_count, class_to_prune)
            to_keep = int(np.ceil(unbalance * max(
                other_class_count)))

            to_delete = all_ids_c[to_keep: len(all_ids_c)]

            self.dataset_x = np.delete(self.dataset_x, to_delete, axis=0)
            self.dataset_y = np.delete(self.dataset_y, to_delete, axis=0)

        # Recount after pruning
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))
        self.per_class_count = per_class_count

        # List of labels
        self.label_table = [str(c) for c in range(10)]

        # Preload all the labels.
        self.labels = self.dataset_y[:]

        # per class ids
        self.per_class_ids = dict()
        ids = np.array(range(len(self.dataset_x)))
        for c in classes:
            self.per_class_ids[c] = ids[self.labels == c]

    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size

        np.random.shuffle(self.per_class_ids[c])
        to_return = self.per_class_ids[c][0:samples]
        return self.dataset_x[to_return]

    def get_label_table(self):
        return self.label_table

    def get_num_classes(self):
        return len( self.label_table )

    def get_class_probability(self):
        return self.per_class_count/sum(self.per_class_count)

    ### ACCESS DATA AND SHAPES ###
    def get_num_samples(self):
        return self.dataset_x.shape[0]

    def get_image_shape(self):
        return [self.dataset_x.shape[1], self.dataset_x.shape[2], self.dataset_x.shape[3]]

    def next_batch(self):
        dataset_x = self.dataset_x
        labels = self.labels

        indices = np.arange(dataset_x.shape[0])

        np.random.shuffle(indices)

        for start_idx in range(0, dataset_x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            access_pattern = sorted(access_pattern)

            yield dataset_x[access_pattern, :, :, :], labels[access_pattern]


import balancing_gan as bagan
from utils import save_image_array

import os



if __name__ == '__main__':
    # Collect arguments
    argParser = OptionParser()
                  
    argParser.add_option("-u", "--unbalance", default=0.2,
                  action="store", type="float", dest="unbalance",
                  help="Unbalance factor u. The minority class has at most u * otherClassSamples instances.")

    argParser.add_option("-s", "--random_seed", default=0,
                  action="store", type="int", dest="seed",
                  help="Random seed for repeatable subsampling.")

    argParser.add_option("-d", "--sampling_mode_for_discriminator", default="uniform",
                  action="store", type="string", dest="dratio_mode",
                  help="Dratio sampling mode (\"uniform\",\"rebalance\").")
    
    argParser.add_option("-g", "--sampling_mode_for_generator", default="uniform",
                  action="store", type="string", dest="gratio_mode",
                  help="Gratio sampling mode (\"uniform\",\"rebalance\").")

    argParser.add_option("-e", "--epochs", default=3,
                  action="store", type="int", dest="epochs",
                  help="Training epochs.")

    argParser.add_option("-l", "--learning_rate", default=0.00005,
                  action="store", type="float", dest="adam_lr",
                  help="Training learning rate.")

    argParser.add_option("-c", "--target_class", default=-1,
                  action="store", type="int", dest="target_class",
                  help="If greater or equal to 0, model trained only for the specified class.")

    (options, args) = argParser.parse_args()

    assert (options.unbalance <= 1.0 and options.unbalance > 0.0), "Data unbalance factor must be > 0 and <= 1"

    print("Executing BAGAN.")

    # Read command line parameters
    np.random.seed(options.seed)
    unbalance = options.unbalance
    gratio_mode = options.gratio_mode
    dratio_mode = options.dratio_mode
    gan_epochs = options.epochs
    adam_lr = options.adam_lr
    opt_class = options.target_class
    batch_size = 32
    dataset_name = 'MNIST'

    # Set channels for mnist.
    channels=1

    # Result directory
    res_dir = "./res_{}_dmode_{}_gmode_{}_unbalance_{}_epochs_{}_lr_{:f}_seed_{}".format(
        dataset_name, dratio_mode, gratio_mode, unbalance, options.epochs, adam_lr, options.seed
    )
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Read initial data.
    print("read input data...")
    bg_train_full = BatchGenerator(BatchGenerator.TRAIN, batch_size,
                                   class_to_prune=None, unbalance=None)
    bg_test = BatchGenerator(BatchGenerator.TEST, batch_size,
                             class_to_prune=None, unbalance=None)

    print("input data loaded...")

    shape = bg_train_full.get_image_shape()

    min_latent_res = shape[-1]
    while min_latent_res > 8:
        min_latent_res = min_latent_res / 2
    min_latent_res = int(min_latent_res)

    classes = bg_train_full.get_label_table()

    # Initialize statistics information
    gan_train_losses = defaultdict(list)
    gan_test_losses = defaultdict(list)

    img_samples = defaultdict(list)

    # For all possible minority classes.
    target_classes = np.array(range(len(classes)))
    if opt_class >= 0:
        min_classes = np.array([opt_class])
    else:
        min_classes = target_classes

    for c in min_classes:
        # If unbalance is 1.0, then the same BAGAN model can be applied to every class because
        # we do not drop any instance at training time.
        if unbalance == 1.0 and c > 0 and (
            os.path.exists("{}/class_0_score.csv".format(res_dir, c)) and
            os.path.exists("{}/class_0_discriminator.h5".format(res_dir, c)) and
            os.path.exists("{}/class_0_generator.h5".format(res_dir, c)) and
            os.path.exists("{}/class_0_reconstructor.h5".format(res_dir, c))
        ):
            # Without additional imbalance, BAGAN does not need to be retrained, we simlink the pregenerated model
            os.symlink("{}/class_0_score.csv".format(res_dir), "{}/class_{}_score.csv".format(res_dir, c))
            os.symlink("{}/class_0_discriminator.h5".format(res_dir), "{}/class_{}_discriminator.h5".format(res_dir, c))
            os.symlink("{}/class_0_generator.h5".format(res_dir), "{}/class_{}_generator.h5".format(res_dir, c))
            os.symlink("{}/class_0_reconstructor.h5".format(res_dir), "{}/class_{}_reconstructor.h5".format(res_dir, c))

        # Unbalance the training set.
        bg_train_partial = BatchGenerator(BatchGenerator.TRAIN, batch_size,
                                          class_to_prune=c, unbalance=unbalance)

        # Train the model (or reload it if already available
        if not (
                os.path.exists("{}/class_{}_score.csv".format(res_dir, c)) and
                os.path.exists("{}/class_{}_discriminator.h5".format(res_dir, c)) and
                os.path.exists("{}/class_{}_generator.h5".format(res_dir, c)) and
                os.path.exists("{}/class_{}_reconstructor.h5".format(res_dir, c))
        ):
            # Training required
            print("Required GAN for class {}".format(c))

            print('Class counters: ', bg_train_partial.per_class_count)

            # Train GAN to balance the data
            gan = bagan.BalancingGAN(
                target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
                adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res
            )
            gan.train(bg_train_partial, bg_test, epochs=gan_epochs)
            gan.save_history(
                res_dir, c
            )

        else:  # GAN pre-trained
            # Unbalance the training.
            print("Loading GAN for class {}".format(c))

            gan = bagan.BalancingGAN(target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
                                     adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res)
            gan.load_models(
                "{}/class_{}_generator.h5".format(res_dir, c),
                "{}/class_{}_discriminator.h5".format(res_dir, c),
                "{}/class_{}_reconstructor.h5".format(res_dir, c),
                bg_train=bg_train_partial  # This is required to initialize the per-class mean and covariance matrix
            )

        # Sample and save images
        img_samples['class_{}'.format(c)] = gan.generate_samples(c=c, samples=10)

        save_image_array(np.array([img_samples['class_{}'.format(c)]]), '{}/plot_class_{}.png'.format(res_dir, c))


