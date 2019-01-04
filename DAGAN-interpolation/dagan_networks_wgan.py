import tensorflow as tf
from dagan_architectures import UResNetGenerator, Discriminator
from utils.interpolations import create_interpolation_interval


class DAGAN:
    def __init__(self, input_x_i, input_x_j, dropout_rate, generator_layer_sizes,
                 discriminator_layer_sizes, generator_layer_padding, z_inputs, batch_size=100, z_dim=100,
                 num_channels=1, is_training=True, augment=True, discr_inner_conv=0, gen_inner_conv=0, num_gpus=1, 
                 use_wide_connections=False):

        """
        Initializes a DAGAN object.
        :param input_x_i: Input image x_i
        :param input_x_j: Input image x_j
        :param dropout_rate: A dropout rate placeholder or a scalar to use throughout the network
        :param generator_layer_sizes: A list with the number of feature maps per layer (generator) e.g. [64, 64, 64, 64]
        :param discriminator_layer_sizes: A list with the number of feature maps per layer (discriminator)
                                                                                                   e.g. [64, 64, 64, 64]
        :param generator_layer_padding: A list with the type of padding per layer (e.g. ["SAME", "SAME", "SAME","SAME"]
        :param z_inputs: A placeholder for the random noise injection vector z (usually gaussian or uniform distribut.)
        :param batch_size: An integer indicating the batch size for the experiment.
        :param z_dim: An integer indicating the dimensionality of the random noise vector (usually 100-dim).
        :param num_channels: Number of image channels
        :param is_training: A boolean placeholder for the training/not training flag
        :param augment: A boolean placeholder that determines whether to augment the data using rotations
        :param discr_inner_conv: Number of inner layers per multi layer in the discriminator
        :param gen_inner_conv: Number of inner layers per multi layer in the generator
        :param num_gpus: Number of GPUs to use for training
        """
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.z_inputs = z_inputs
        self.num_gpus = num_gpus

        self.g = UResNetGenerator(batch_size=self.batch_size, layer_sizes=generator_layer_sizes,
                                  num_channels=num_channels, layer_padding=generator_layer_padding,
                                  inner_layers=gen_inner_conv, name="generator")

        self.d = Discriminator(batch_size=self.batch_size, layer_sizes=discriminator_layer_sizes,
                               inner_layers=discr_inner_conv, use_wide_connections=use_wide_connections, name="discriminator")

        self.input_x_i = input_x_i
        self.input_x_j = input_x_j
        self.dropout_rate = dropout_rate
        self.training_phase = is_training
        self.augment = augment

    def rotate_data(self, image_a, image_b):
        """
        Rotate 2 images by the same number of degrees
        :param image_a: An image a to rotate k degrees
        :param image_b: An image b to rotate k degrees
        :return: Two images rotated by the same amount of degrees
        """
        random_variable = tf.unstack(tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32, seed=None, name=None))
        image_a = tf.image.rot90(image_a, k=random_variable[0])
        image_b = tf.image.rot90(image_b, k=random_variable[0])
        return [image_a, image_b]

    def rotate_batch(self, batch_images_a, batch_images_b):
        """
        Rotate two batches such that every element from set a with the same index as an element from set b are rotated
        by an equal amount of degrees
        :param batch_images_a: A batch of images to be rotated
        :param batch_images_b: A batch of images to be rotated
        :return: A batch of images that are rotated by an element-wise equal amount of k degrees
        """
        shapes = map(int, list(batch_images_a.get_shape()))
        batch_size, x, y, c = shapes
        with tf.name_scope('augment'):
            batch_images_unpacked_a = tf.unstack(batch_images_a)
            batch_images_unpacked_b = tf.unstack(batch_images_b)
            new_images_a = []
            new_images_b = []
            for image_a, image_b in zip(batch_images_unpacked_a, batch_images_unpacked_b):
                rotate_a, rotate_b = self.augment_rotate(image_a, image_b)
                new_images_a.append(rotate_a)
                new_images_b.append(rotate_b)

            new_images_a = tf.stack(new_images_a)
            new_images_a = tf.reshape(new_images_a, (batch_size, x, y, c))
            new_images_b = tf.stack(new_images_b)
            new_images_b = tf.reshape(new_images_b, (batch_size, x, y, c))
            return [new_images_a, new_images_b]

    def generate(self, conditional_images, z_input=None):
        """
        Generate samples with the DAGAN
        :param conditional_images: Images to condition DAGAN on.
        :param z_input: Random noise to condition the DAGAN on. If none is used then the method will generate random
        noise with dimensionality [batch_size, z_dim]
        :return: A batch of generated images, one per conditional image
        """
        if z_input is None:
            z_input = tf.random_normal([self.batch_size, self.z_dim], mean=0, stddev=1)

        generated_samples, encoder_layers, decoder_layers = self.g(z_input,
                               conditional_images,
                               training=self.training_phase,
                               dropout_rate=self.dropout_rate)
        return generated_samples

    def augment_rotate(self, image_a, image_b):
        r = tf.unstack(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32, seed=None, name=None))
        rotate_boolean = tf.equal(0, r, name="check-rotate-boolean")
        [image_a, image_b] = tf.cond(rotate_boolean[0], lambda: self.rotate_data(image_a, image_b),
                        lambda: [image_a, image_b])
        return image_a, image_b

    def data_augment_batch(self, batch_images_a, batch_images_b):
        """
        Apply data augmentation to a set of image batches if self.augment is set to true
        :param batch_images_a: A batch of images to augment
        :param batch_images_b: A batch of images to augment
        :return: A list of two augmented image batches
        """
        [images_a, images_b] = tf.cond(self.augment, lambda: self.rotate_batch(batch_images_a, batch_images_b),
                                       lambda: [batch_images_a, batch_images_b])
        return images_a, images_b

    def save_features(self, name, features):
        """
        Save feature activations from a network
        :param name: A name for the summary of the features
        :param features: The features to save
        """
        for i in range(len(features)):
            shape_in = features[i].get_shape().as_list()
            channels = shape_in[3]
            y_channels = 8
            x_channels = channels / y_channels

            activations_features = tf.reshape(features[i], shape=(shape_in[0], shape_in[1], shape_in[2],
                                                                        y_channels, x_channels))

            activations_features = tf.unstack(activations_features, axis=4)
            activations_features = tf.concat(activations_features, axis=2)
            activations_features = tf.unstack(activations_features, axis=3)
            activations_features = tf.concat(activations_features, axis=1)
            activations_features = tf.expand_dims(activations_features, axis=3)
            tf.summary.image('{}_{}'.format(name, i), activations_features)

    def loss(self, gpu_id):

        """
        Builds models, calculates losses, saves tensorboard information.
        :param gpu_id: The GPU ID to calculate losses for.
        :return: Returns the generator and discriminator losses.
        """
        with tf.name_scope("losses_{}".format(gpu_id)):

            input_a, input_b = self.data_augment_batch(self.input_x_i[gpu_id], self.input_x_j[gpu_id])
            x_g = self.generate(input_a)

            g_same_class_outputs, g_discr_features = self.d(x_g, input_a, training=self.training_phase,
                                          dropout_rate=self.dropout_rate)

            t_same_class_outputs, t_discr_features, t_lastlayer_features = self.d(input_b, input_a,
                                                                                  training=self.training_phase,
                                          dropout_rate=self.dropout_rate, extended_output=True)

            # Remove comments to save discriminator feature activations
            # self.save_features(name="generated_discr_layers", features=g_discr_features)
            # self.save_features(name="real_discr_layers", features=t_discr_features)

            d_real = t_same_class_outputs
            d_fake = g_same_class_outputs
            d_loss_GAN = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            d_loss = d_loss_GAN
            g_loss = -tf.reduce_mean(d_fake)

            alpha = tf.random_uniform(
                shape=[self.batch_size, 1],
                minval=0.,
                maxval=1.
            )
            input_shape = input_a.get_shape()
            input_shape = [int(n) for n in input_shape]
            differences_g = x_g - input_b
            differences_g = tf.reshape(differences_g, (self.batch_size, input_shape[1]*input_shape[2]*input_shape[3]))
            interpolates_g = input_b + tf.reshape(alpha * differences_g, (self.batch_size, input_shape[1],
                                                                          input_shape[2], input_shape[3]))
            pre_grads, grad_features = self.d(interpolates_g, input_a, dropout_rate=self.dropout_rate,
                                     training=self.training_phase)
            gradients = tf.gradients(pre_grads, [interpolates_g, input_a])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_summary_d = tf.reduce_mean(slopes)
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            d_loss += 10 * gradient_penalty

            #improving improved WGAN training
            d1, _, d1_features = self.d(input_b, input_a, training=self.training_phase, dropout_rate=self.dropout_rate,
                                        extended_output=True)
            # for d2, t_same_class_outputs is reused, as it is also computed with dropout
            distance_outputs = tf.abs(t_same_class_outputs - d1)
            distance_features = tf.sqrt(tf.reduce_sum(tf.square(t_lastlayer_features-d1_features), axis=1))
            CT = tf.reduce_mean(distance_outputs + 0.1*distance_features)
            d_loss += 2*CT
            tf.check_numerics(d_loss, 'd_loss_contains_nan')

            #summaries

            tf.add_to_collection('g_losses', g_loss)
            tf.add_to_collection('d_losses', d_loss)
            tf.summary.scalar('g_losses', g_loss)
            tf.summary.scalar('d_losses', d_loss)
            tf.summary.scalar('d_gradients', gradient_summary_d)
            tf.summary.histogram('d_slopes_hist', slopes)
            tf.summary.scalar('gradient_penalty', gradient_penalty)
            tf.summary.scalar('consistency_term', CT)
            tf.summary.scalar('d_loss_gan', d_loss_GAN)

            tf.summary.scalar('d_loss_real', tf.reduce_mean(d_real))
            tf.summary.scalar('d_loss_fake', tf.reduce_mean(d_fake))
            tf.summary.image('output_generated_images', [tf.concat(tf.unstack(x_g, axis=0), axis=0)])
            # tf.summary.image('output_input_a', [tf.concat(tf.unstack(input_a, axis=0), axis=0)])
            # tf.summary.image('output_input_b', [tf.concat(tf.unstack(input_b, axis=0), axis=0)])

        return {
            "g_losses": tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            "d_losses": tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')
        }

    def train(self, opts, losses):

        """
        Returns ops for training our DAGAN system.
        :param opts: A dict with optimizers.
        :param losses: A dict with losses.
        :return: A dict with training ops for the dicriminator and the generator.
        """
        opt_ops = dict()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients_g = opts["g_opt"].compute_gradients(losses["g_losses"],
                                                          var_list=self.g.variables,
                                                          colocate_gradients_with_ops=True)
            opt_ops["g_opt_op"] = opts["g_opt"].apply_gradients(gradients_g)
            # opt_ops["g_opt_op"] = opts["g_opt"].minimize(losses["g_losses"],
            #                               var_list=self.g.variables,
            #                               colocate_gradients_with_ops=True)
            gradients_d = opts["d_opt"].compute_gradients(losses["d_losses"],
                                                         var_list=self.d.variables,
                                                         colocate_gradients_with_ops=True)
            opt_ops["d_opt_op"] = opts["d_opt"].apply_gradients(gradients_d)
            # opt_ops["d_opt_op"] = opts["d_opt"].minimize(losses["d_losses"],
            #                                              var_list=self.d.variables,
            #                                              colocate_gradients_with_ops=True)

            # Save gradients in summary:
            for g, v in gradients_g:
                if g is not None:
                    tf.summary.histogram("{}/grad_g/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad_g/sparsity".format(v.name), tf.nn.zero_fraction(g))

            for g, v in gradients_d:
                if g is not None:
                    tf.summary.histogram("{}/grad_d/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad_d/sparsity".format(v.name), tf.nn.zero_fraction(g))

        return opt_ops

    def init_train(self, learning_rate=1e-7, beta1=0.0, beta2=0.9):
        """
        Initialize training by constructing the summary, loss and ops
        :param learning_rate: The learning rate for the Adam optimizer
        :param beta1: Beta1 for the Adam optimizer
        :param beta2: Beta2 for the Adam optimizer
        :return: summary op, losses and training ops.
        """

        losses = dict()
        opts = dict()

        if self.num_gpus > 0:
            device_ids = ['/gpu:{}'.format(i) for i in range(self.num_gpus)]
        else:
            device_ids = ['/cpu:0']
        for gpu_id, device_id in enumerate(device_ids):
            with tf.device(device_id):
                total_losses = self.loss(gpu_id=gpu_id)
                for key, value in total_losses.items():
                    if key not in losses.keys():
                        losses[key] = [value]
                    else:
                        losses[key].append(value)

        for key in list(losses.keys()):
            losses[key] = tf.reduce_mean(losses[key], axis=0)
            opts[key.replace("losses", "opt")] = tf.train.AdamOptimizer(beta1=beta1, beta2=beta2,
                                                                            learning_rate=learning_rate)

        apply_grads_ops = self.train(opts=opts, losses=losses)
        summary = tf.summary.merge_all()

        return summary, losses, apply_grads_ops

    def sample_same_images(self):
        """
        Samples images from the DAGAN using input_x_i as image conditional input and z_inputs as the gaussian noise.
        :return: Inputs and generated images
        """
        conditional_inputs = self.input_x_i[0]
        generated = self.generate(conditional_inputs,
           z_input=self.z_inputs)

        return self.input_x_i[0], generated

    def generate_multi_batch(self):
        generated_multi_batch = []
        for gpu in range(self.num_gpus):
            self.z_inputs = tf.random_normal()
            generated_batch = self.generate(self.input_x_i[gpu], z_input=self.z_inputs)
            generated_multi_batch.append(generated_batch)
        generated_multi_batch = tf.stack(generated_multi_batch)
        return generated_multi_batch


    def encode(self, conditional_images):
        """
        encode images to latent space with the DAGAN
        :param conditional_images: Images to condition DAGAN on.
        noise with dimensionality [batch_size, z_dim]
        :return: A batch of encoded latent spaces, one per conditional image
        """

        z_input = tf.zeros([self.batch_size, self.z_dim])
        _, _, _, g_conv_encoder = self.g(z_input,
                                         conditional_images,
                                         training=self.training_phase,
                                         dropout_rate=self.dropout_rate,
                                         output_latent_space=True)
        return g_conv_encoder

    def interpolate_inter_class(self, images_class1, images_class2):
        """ Create a batch of interpolated pictures between the first image of images_class1 and the first image
        of images_class2"""
        z_input = tf.zeros([self.batch_size, self.z_dim])
        latent_vectors_class1 = self.encode(images_class1)
        latent_vectors_class2 = self.encode(images_class2)
        # take the first latent vector of each class and interpolate between the two (number of interpolations
        # is equal to the batch size
        interpolated_latent_batch = create_interpolation_interval(latent_vectors_class1[0], latent_vectors_class2[0], self.batch_size)

        generated_samples, _, _ = self.g(z_input, images_class1,
                                         training=self.training_phase,
                                         dropout_rate=self.dropout_rate,
                                         generate_from_latent_space=True,
                                         latent_inputs=interpolated_latent_batch)
        new_shape = [1,images_class1.shape[1],images_class1.shape[2],images_class1.shape[3]]
        generated_samples = tf.concat([tf.reshape(images_class1[0],new_shape),generated_samples,
                                       tf.reshape(images_class2[0],new_shape)],axis=0)
        return generated_samples

    def interpolate_intra_class(self,images):
        """ Create a batch of interpolated pictures between the first two images of the given batch """
        z_input = tf.zeros([self.batch_size, self.z_dim])
        latent_vectors = self.encode(images)
        # take the first two latent vectors and interpolate between the two (number of interpolations
        # is equal to the batch size
        interpolated_latent_batch = create_interpolation_interval(latent_vectors[0], latent_vectors[1], self.batch_size)

        generated_samples, _, _ = self.g(z_input, images,
                                         training=self.training_phase,
                                         dropout_rate=self.dropout_rate,
                                         generate_from_latent_space=True,
                                         latent_inputs=interpolated_latent_batch)
        new_shape = [1, images.shape[1], images.shape[2], images.shape[3]]
        generated_samples = tf.concat([tf.reshape(images[0], new_shape), generated_samples,
                                       tf.reshape(images[1], new_shape)], axis=0)
        return generated_samples


