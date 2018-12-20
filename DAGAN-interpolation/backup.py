def __call__(self, z_inputs, conditional_input, training=False, dropout_rate=0.0, output_latent_space=False,
             generate_from_latent_space=False, latent_inputs=None):
    """
    Apply network on data.
    :param z_inputs: Random noise to inject [batch_size, z_dim]
    :param conditional_input: A batch of images to use as conditionals [batch_size, height, width, channels]
    :param training: Training placeholder or boolean
    :param dropout_rate: Dropout rate placeholder or float
    :return: Returns x_g (generated images), encoder_layers(encoder features), decoder_layers(decoder features)
    """
    conditional_input = tf.convert_to_tensor(conditional_input)
    with tf.variable_scope(self.name, reuse=self.reuse):
        # reshape from inputs
        outputs = conditional_input
        encoder_layers = []
        current_layers = [outputs]
        with tf.variable_scope('conv_layers'):

            for i, layer_size in enumerate(self.layer_sizes):
                encoder_inner_layers = [outputs]
                with tf.variable_scope('g_conv{}'.format(i)):
                    if i == 0:  # first layer is a single conv layer instead of MultiLayer for best results
                        outputs = self.conv_layer(outputs, num_filters=64,
                                                  filter_size=(3, 3), strides=(2, 2))
                        outputs = leaky_relu(features=outputs)
                        outputs = batch_norm(outputs, decay=0.99, scale=True,
                                             center=True, is_training=training,
                                             renorm=True)
                        current_layers.append(outputs)
                        encoder_inner_layers.append(outputs)
                    else:
                        for j in range(self.inner_layers[i]):  # Build the inner Layers of the MultiLayer
                            outputs = self.add_encoder_layer(input=outputs,
                                                             training=training,
                                                             name="encoder_layer_{}_{}".format(i, j),
                                                             layer_to_skip_connect=current_layers,
                                                             num_features=self.layer_sizes[i],
                                                             dim_reduce=False,
                                                             local_inner_layers=encoder_inner_layers,
                                                             dropout_rate=dropout_rate)
                            encoder_inner_layers.append(outputs)
                            current_layers.append(outputs)
                        # add final dim reducing conv layer for this MultiLayer
                        outputs = self.add_encoder_layer(input=outputs, name="encoder_layer_{}".format(i),
                                                         training=training, layer_to_skip_connect=current_layers,
                                                         local_inner_layers=encoder_inner_layers,
                                                         num_features=self.layer_sizes[i],
                                                         dim_reduce=True, dropout_rate=dropout_rate)
                        current_layers.append(outputs)
                    encoder_layers.append(outputs)

        if generate_from_latent_space:
            g_conv_encoder = latent_inputs
        else:
            g_conv_encoder = outputs

        with tf.variable_scope("vector_expansion"):  # Used for expanding the z injected noise to match the
            # dimensionality of the various decoder MultiLayers, injecting
            # noise into multiple decoder layers in a skip-connection way
            # improves quality of results. We inject in the first 3 decode
            # multi layers
            num_filters = 8
            z_layers = []
            concat_shape = [layer_shape.get_shape().as_list() for layer_shape in encoder_layers]

            for i in range(len(self.inner_layers)):
                h = concat_shape[len(encoder_layers) - 1 - i][1]
                w = concat_shape[len(encoder_layers) - 1 - i][1]
                z_dense = tf.layers.dense(z_inputs, h * w * num_filters)
                z_reshape_noise = tf.reshape(z_dense, [self.batch_size, h, w, num_filters])
                num_filters /= 2
                num_filters = int(num_filters)
                print(z_reshape_noise)
                z_layers.append(z_reshape_noise)

        outputs = g_conv_encoder
        decoder_layers = []
        current_layers = [outputs]
        with tf.variable_scope('g_deconv_layers'):
            for i in range(len(self.layer_sizes) + 1):
                if i < 3:  # Pass the injected noise to the first 3 decoder layers for sharper results
                    outputs = tf.concat([z_layers[i], outputs], axis=3)
                    current_layers[-1] = outputs
                idx = len(self.layer_sizes) - 1 - i
                num_features = self.layer_sizes[idx]
                inner_layers = self.inner_layers[idx]
                upscale_shape = encoder_layers[idx].get_shape().as_list()
                if idx < 0:
                    num_features = self.layer_sizes[0]
                    inner_layers = self.inner_layers[0]
                    outputs = tf.concat([outputs, conditional_input], axis=3)
                    upscale_shape = conditional_input.get_shape().as_list()

                with tf.variable_scope('g_deconv{}'.format(i)):
                    decoder_inner_layers = [outputs]
                    for j in range(inner_layers):
                        if i == 0 and j == 0:
                            outputs = self.add_decoder_layer(input=outputs,
                                                             name="decoder_inner_conv_{}_{}"
                                                             .format(i, j),
                                                             training=training,
                                                             layer_to_skip_connect=current_layers,
                                                             num_features=num_features,
                                                             dim_upscale=False,
                                                             local_inner_layers=decoder_inner_layers,
                                                             dropout_rate=dropout_rate)
                            decoder_inner_layers.append(outputs)
                        else:
                            outputs = self.add_decoder_layer(input=outputs,
                                                             name="decoder_inner_conv_{}_{}"
                                                             .format(i, j), training=training,
                                                             layer_to_skip_connect=current_layers,
                                                             num_features=num_features,
                                                             dim_upscale=False,
                                                             # todo: I changed this from false to true
                                                             local_inner_layers=decoder_inner_layers,
                                                             w_size=upscale_shape[1],
                                                             h_size=upscale_shape[2],
                                                             dropout_rate=dropout_rate)
                            decoder_inner_layers.append(outputs)
                    current_layers.append(outputs)
                    decoder_layers.append(outputs)

                    if idx >= 0:
                        upscale_shape = encoder_layers[idx - 1].get_shape().as_list()
                        if idx == 0:
                            upscale_shape = conditional_input.get_shape().as_list()
                        outputs = self.add_decoder_layer(
                            input=outputs,
                            name="decoder_outer_conv_{}".format(i),
                            training=training,
                            layer_to_skip_connect=current_layers,
                            num_features=num_features,
                            dim_upscale=True, local_inner_layers=decoder_inner_layers, w_size=upscale_shape[1],
                            h_size=upscale_shape[2], dropout_rate=dropout_rate)
                        current_layers.append(outputs)
                    if (idx - 1) >= 0:
                        outputs = tf.concat([outputs, encoder_layers[idx - 1]], axis=3)
                        current_layers[-1] = outputs

            high_res_layers = []

            for p in range(2):
                outputs = self.conv_layer(outputs, self.layer_sizes[0], [3, 3], strides=(1, 1),
                                          transpose=False)
                outputs = leaky_relu(features=outputs)

                outputs = batch_norm(outputs,
                                     decay=0.99, scale=True,
                                     center=True, is_training=training,
                                     renorm=True)
                high_res_layers.append(outputs)
            outputs = self.conv_layer(outputs, self.num_channels, [3, 3], strides=(1, 1),
                                      transpose=False)
        # output images
        with tf.variable_scope('g_tanh'):
            gan_decoder = tf.tanh(outputs, name='outputs')

    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    if self.build:
        print("generator_total_layers", self.conv_layer_num)
        count_parameters(self.variables, name="generator_parameter_num")
    self.build = False

    if output_latent_space:
        return gan_decoder, encoder_layers, decoder_layers, g_conv_encoder
    else:
        return gan_decoder, encoder_layers, decoder_layers
