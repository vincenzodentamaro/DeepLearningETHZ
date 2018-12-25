import utils.interpolations
import tqdm
from utils.storage import *
from tensorflow.contrib import slim

from dagan_networks_wgan import *
from utils.sampling import *


class ExperimentBuilder(object):
    def __init__(self, parser, data):
        tf.reset_default_graph()

        args = parser.parse_args()
        self.continue_from_epoch = args.continue_from_epoch
        self.experiment_name = args.experiment_title
        self.saved_models_filepath, self.log_path, self.save_image_path, self.generations_path\
            = build_experiment_folder(self.experiment_name)
        self.num_gpus = args.num_of_gpus
        self.batch_size = args.batch_size
        gen_depth_per_layer = args.generator_inner_layers
        discr_depth_per_layer = args.discriminator_inner_layers
        self.z_dim = args.z_dim
        self.num_generations = args.num_generations
        self.dropout_rate_value = args.dropout_rate_value
        self.data = data
        self.reverse_channels = False

        generator_layers = [64, 64, 128, 128]
        discriminator_layers = [64, 64, 128, 128]

        gen_inner_layers = [gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer, gen_depth_per_layer]
        discr_inner_layers = [discr_depth_per_layer, discr_depth_per_layer, discr_depth_per_layer,
                              discr_depth_per_layer]
        generator_layer_padding = ["SAME", "SAME", "SAME", "SAME"]

        image_height = data.image_height
        image_width = data.image_width
        image_channels = data.image_channels

        self.input_x_i = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channels], 'inputs-1')
        self.input_x_j = tf.placeholder(tf.float32, [self.num_gpus, self.batch_size, image_height, image_width,
                                                     image_channels], 'inputs-2-same-class')

        self.z_input = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], 'z-input')
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.random_rotate = tf.placeholder(tf.bool, name='rotation-flag')
        self.dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')

        dagan = DAGAN(batch_size=self.batch_size, input_x_i=self.input_x_i, input_x_j=self.input_x_j,
                      dropout_rate=self.dropout_rate, generator_layer_sizes=generator_layers,
                      generator_layer_padding=generator_layer_padding, num_channels=data.x.shape[4],
                      is_training=self.training_phase, augment=self.random_rotate,
                      discriminator_layer_sizes=discriminator_layers, discr_inner_conv=discr_inner_layers,
                      gen_inner_conv=gen_inner_layers, num_gpus=self.num_gpus, z_dim=self.z_dim, z_inputs=self.z_input)

        self.generated_multi_batch = dagan.generate_multi_batch()
        self.total_gen_batches = int(data.generation_data_size / (self.batch_size * self.num_gpus))
        self.init = tf.global_variables_initializer()
        self.spherical_interpolation = True

        if self.continue_from_epoch == -1:
            save_statistics(self.log_path, ["epoch", "total_d_loss", "total_g_loss", "total_d_val_loss",
                                              "total_g_val_loss"], create=True)

    def run_experiment(self):
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True)) as sess:
            sess.run(self.init)
            self.writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            self.saver = tf.train.Saver()
            if self.continue_from_epoch != -1:
                checkpoint = "{}/{}_{}.ckpt".format(self.saved_models_filepath, self.experiment_name,
                                                    self.continue_from_epoch)
                variables_to_restore = []
                for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    print(var)
                    variables_to_restore.append(var)

                tf.logging.info('Fine-tuning from %s' % checkpoint)

                fine_tune = slim.assign_from_checkpoint_fn(
                    checkpoint,
                    variables_to_restore,
                    ignore_missing_vars=True)
                fine_tune(sess)


            with tqdm.tqdm(total=self.total_gen_batches) as pbar_samp:
                info_file_gen = None
                for i in range(self.total_gen_batches):
                    x_gen_a, info_file_a = self.data.get_multi_batch()
                    info_file_gen = generate_sample_batch(sess=sess, generated_multi_batch=self.generated_multi_batch,
                                                          inputs=x_gen_a,
                                          input_a=self.input_x_i, dropout_rate=self.dropout_rate,
                                          dropout_rate_value=self.dropout_rate_value, data=self.data,
                                          batch_size=self.batch_size, file_prefix=self.generations_path,
                                          info_file_batch=info_file_a, info_file_gen=info_file_gen,
                                                          training_phase=False, z_input = self.z_input)

                    pbar_samp.update(1)
