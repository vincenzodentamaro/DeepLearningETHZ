import data as dataset
from experiment_builder import ExperimentBuilder
from utils.parser_util import ArgsMimicker
batch_size = 32
generator_inner_layers = 3
discriminator_inner_layers = 5
num_generations = 64
experiment_title = 'omniglot_dagan_experiment_default'
num_of_gpus = 1
z_dim = 100
dropout_rate_value = 0.5
continue_from_epoch = -1
use_wide_connections = False

args = ArgsMimicker(batch_size=batch_size, generator_inner_layers=generator_inner_layers,
                    discriminator_inner_layers=discriminator_inner_layers, num_generations=num_generations,
                    experiment_title=experiment_title, num_of_gpus=num_of_gpus, z_dim=z_dim,
                    dropout_rate_value=dropout_rate_value, continue_from_epoch=continue_from_epoch,
                    use_wide_connections=use_wide_connections)




#set the data provider to use for the experiment
data = dataset.OmniglotDAGANDataset(batch_size=batch_size, last_training_class_index=900, reverse_channels=True,
                                    num_of_gpus=num_of_gpus, gen_batches=10)
#init experiment
experiment = ExperimentBuilder(args, data=data)
#run experiment
experiment.run_experiment()