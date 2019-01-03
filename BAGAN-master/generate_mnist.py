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

import balancing_gan as bagan
from rw.batch_generator_mnist import MnistBatchGenerator as BatchGenerator
from utils import save_image_array

import os
if __name__ == '__main__':
    # Collect arguments
    argParser = OptionParser()
                  
    argParser.add_option("-u", "--unbalance", default=0.2,
                  action="store", type="float", dest="unbalance",
                  help="Unbalance factor u. The minority class has at most u * otherClassSamples instances.")
    argParser.add_option("-a", "--amount", default=10000,
                  action="store", type="float", dest="amount",
                  help="Amount of training samples used from MNIST.")

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
    amount=options.amount
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
    res_dir = "./res_{}_dmode_{}_gmode_{}_unbalance_{}_epochs_{}_lr_{:f}_seed_{}".format(dataset_name, dratio_mode, gratio_mode, unbalance, options.epochs, adam_lr, options.seed)
    for c in range(0, 10):
        print("Loading GAN for class {}".format(c))
        bg_train_partial = BatchGenerator(BatchGenerator.TRAIN, batch_size, class_to_prune=c, unbalance=unbalance)
        gan = bagan.BalancingGAN(target_classes, c, dratio_mode=dratio_mode, gratio_mode=gratio_mode, adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=7)
        gan.load_models("{}/class_0_generator.h5".format(res_dir),"{}/class_0_discriminator.h5".format(res_dir),"{}/class_0_reconstructor.h5".format(res_dir),bg_train=bg_train_partial)

        # Sample and save images
        img_samples['class_{}'.format(c)] = gan.generate_samples(c=c, samples=300)
        np.save('{}/samples_class_{}.npy'.format(res_dir,c),img_samples['class_{}'.format(c)])
        save_image_array(np.array([img_samples['class_{}'.format(c)]]), '{}/plot_class_{}.png'.format(res_dir, c))
