import argparse


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def get_args():
    parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
    parser.add_argument('--batch_size', nargs="?", type=int, default=8, help='batch_size for experiment')
    parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1,
                        help='Number of inner layers per multi layer in the discriminator')
    parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=1,
                        help='Number of inner layers per multi layer in the generator')
    parser.add_argument('--experiment_title', nargs="?", type=str, default="omniglot",
                        help='Experiment name')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1,
                        help='continue from checkpoint of epoch')
    parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
    parser.add_argument('--dropout_rate_value', type=float, default=0.5,
                        help='A dropout rate placeholder or a scalar to use throughout the network')
    parser.add_argument('--num_generations', nargs="?", type=int, default=8,
                        help='The number of samples generated for use in the spherical interpolations at the end of '
                             'each epoch')
    parser.add_argument('--use_wide_connections', nargs="?", type=str, default="False",
                        help='Whether to use wide connections in discriminator')
    parser.add_argument('--infofile', type=str, default='../data_info_files/final_train_info.csv')
    parser.add_argument('--data_dir', type=str, default='../train_reduced')
    args = parser.parse_args()
    batch_size = args.batch_size
    num_gpus = args.num_of_gpus

    args_dict = vars(args)
    for key in list(args_dict.keys()):
        print(key, args_dict[key])

        if args_dict[key] == "True":
            args_dict[key] = True
        elif args_dict[key] == "False":
            args_dict[key] = False
    args = Bunch(args_dict)

    return batch_size, num_gpus, args


class ArgsMimicker(object):
    def __init__(self, batch_size,generator_inner_layers, discriminator_inner_layers, num_generations, experiment_title,
                 num_of_gpus, z_dim, dropout_rate_value,continue_from_epoch=-1, use_wide_connections=False):
        self.generator_inner_layers = generator_inner_layers
        self.discriminator_inner_layers = discriminator_inner_layers
        self.num_generations = num_generations
        self.experiment_title = experiment_title
        self.num_of_gpus = num_of_gpus
        self.z_dim = z_dim
        self.dropout_rate_value = dropout_rate_value
        self.continue_from_epoch = continue_from_epoch
        self.use_wide_connections = use_wide_connections
        self.batch_size = batch_size
