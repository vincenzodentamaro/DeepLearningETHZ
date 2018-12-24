import argparse
import data as dataset
from generation_builder import ExperimentBuilder

parser = argparse.ArgumentParser(description='Welcome to GAN-Shot-Learning script')
parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='batch_size for experiment')
parser.add_argument('--discriminator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--generator_inner_layers', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--experiment_title', nargs="?", type=str, default="densenet_generator_fc", help='Experiment name')
parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='continue from checkpoint of epoch')
parser.add_argument('--num_of_gpus', nargs="?", type=int, default=1, help='discr_number_of_conv_per_layer')
parser.add_argument('--z_dim', nargs="?", type=int, default=100, help='The dimensionality of the z input')
parser.add_argument('--dropout_rate_value', type=float, default=0.5, help='dropout_rate_value')
parser.add_argument('--num_generations', nargs="?", type=int, default=64, help='num_generations')
parser.add_argument('--gen_labels', nargs="?", default=None, help='labels to generate')
parser.add_argument('--infofile', type=str, default='../data_info_files/final_train_info.csv')
parser.add_argument('--data_dir', type=str, default='../train_reduced')


args = parser.parse_args()
batch_size = args.batch_size
num_gpus = args.num_of_gpus

data = dataset.GeneratePaintingsDataset(batch_size=batch_size, reverse_channels=False,num_of_gpus = args.num_of_gpus,
                                        gen_labels=args.gen_labels, csv_file=args.infofile, root_dir=args.data_dir)
experiment = ExperimentBuilder(parser, data=data)
experiment.run_experiment()