import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import datetime
import time
import sys
import os
import numpy as np

# ==================== Parameters =====================================

# Model Hyperparameters
tf.flags.DEFINE_integer("num_filters_first_layer", 32, "Number of filters per filter size for first layer(default: 32)")
tf.flags.DEFINE_integer("num_filters_second_layer", 64, "Number of filters per filter size for 2nd layer (default: 64)")
tf.flags.DEFINE_integer("patch_size", 5, "Size of the filter (default: 5)")
tf.flags.DEFINE_integer("size_fully_connected_layer", 512, "Size of the fully connected layer (default: 1024)")
tf.flags.DEFINE_float("keep_prob", 0.8, "dropout strength on fully connected layer")
tf.flags.DEFINE_float("lambda_reg", 0.001, "Regularization strength for L2 regularizer")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 3000, "Number of training epochs (default: 2000)")
tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")


# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("run_name", None, "Suffix for output directory. If None, a timestamp is used instead")


#New parameters for the task at hand
tf.flags.DEFINE_integer("amount", 1000, "Amount of training samples (default: 1000)")
tf.flags.DEFINE_integer("augmentation", 0, "add BAGAN generated data: 0 for false; 1 for a 1000 BAGAN generated data made from the same 1000 sample; 2 for a 1000 BAGAN generated data from a random 1000 samples; 3 for a 1000 BAGAN generated data from the whole set of MNIST; 4 for adding the same 1000 samples but with noise (classical data augmentation).")
tf.flags.DEFINE_integer("fancy_CNN", 1, "Use fancy CNN or not: 0 for false, 1 for true ")
tf.flags.DEFINE_integer("easy_task", 0, "Do easy task (classify 0, 1 from the rest): 0 for false, 1 for true ")


FLAGS = tf.flags.FLAGS
#FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
easy_task=FLAGS.easy_task
lambda_regs=FLAGS.lambda_reg
keep_prob=FLAGS.keep_prob
size_fully_connected_layer=FLAGS.size_fully_connected_layer
num_filters_first_layer=FLAGS.num_filters_first_layer
num_filters_second_layer=FLAGS.num_filters_second_layer

#we have 2 CNN's, one "fancy" taken from Deep Learning course Exercise 5, and one basic with very limited regularization, 20 nodes, and only 6 filters.
if FLAGS.fancy_CNN==0:
    from rw.classifier import BASIC_CNN as C
    lambda_regs=0.001
    keep_prob=0.95
    size_fully_connected_layer=20
    num_filters_first_layer=2
    num_filters_second_layer=4
else:
    from rw.classifier import CNN as C




#==================== Data Loading and Preparation ===================

# downloads the MNIST data if it doesn't exist
# each image is of size 28x28, and is stored in a flattened version (size 784)
# the label for each image is a one-hot vector (size 10)
# the data is divided in training set (mnist.train) of size 55,000, validation set
# (mnist.validation) of size 5,000 and test set (mnist.test) of size 10,000
# for each set the images and labels are given (e.g. mnist.train.images of size
# [55,000, 784] and mnist.train.labels of size [55,000, 10])

amount=FLAGS.amount
augmentation=FLAGS.augmentation
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#import only a limited amount of data
dataset_x_train = mnist.train.images[0:amount]
dataset_y_train = mnist.train.labels[0:amount]

#depending on what augmentation option we choose, import different BAGAN generated images
if augmentation==1 or augmentation==2 or augmentation==3:
    if augmentation==1:
        samples='GAN-generated_DATA/1000_class_'
    if augmentation==2:
        samples='GAN-generated_DATA/samples_class_'
    if augmentation==3:
        samples='GAN-generated_DATA/50000samples_class_'
    #formatting the BAGAN generated data in the right format
    dataset_x_train_aug=np.load(samples+'0.npy')
    dataset_x_train_aug=np.reshape(dataset_x_train,(amount,784))
    dataset_x_train_aug=dataset_x_train_aug[0:int(amount/10)]

    temp=np.full(int(amount/10), 0)
    z=np.zeros((int(amount/10),10))
    z[np.arange(int(amount/10)), temp] = 1
    for i in range(1,10):
        temp=np.load(samples+str(i)+'.npy')
        temp=np.reshape(temp,(amount,784))
        temp=temp[0:int(amount/10)]
        
        temp3=np.full(int(amount/10),i)
        temp1=np.zeros((int(amount/10), 10))
        temp1[np.arange(int(amount/10)), temp3] = 1
        dataset_x_train_aug=np.concatenate((dataset_x_train_aug,temp), axis=0)
        z=np.concatenate((z,temp1), axis=0)

    #adding the normal MNIST data to the BAGAN generated data
    dataset_x_train=np.concatenate((dataset_x_train,dataset_x_train_aug),axis=0)
    dataset_y_train=np.concatenate((dataset_y_train,z),axis=0)
    dataset_y_train_aug=z
#add the same 1000 samples but with noise
if augmentation==4:
    dataset_x_train_aug = mnist.train.images[0:amount]
    dataset_y_train_aug = mnist.train.labels[0:amount]
    for i in range(0,len(dataset_x_train_aug)):
        dataset_x_train_aug[i]=dataset_x_train_aug[i]+np.random.normal(loc=0.0, scale=0.005, size=784)
    dataset_x_train=np.concatenate((dataset_x_train,dataset_x_train_aug),axis=0)
    dataset_y_train=np.concatenate((dataset_y_train,dataset_y_train_aug),axis=0)


    
#shuffle data before training    
p = np.random.permutation(len(dataset_x_train))
dataset_x_train=dataset_x_train[p]
dataset_y_train=dataset_y_train[p]      

#the easy task is classifying 0 from 1 from the rest, so we must change the training set accordingly
if easy_task==1:
    for i in range(0,len(dataset_y_train)):
        for j in range(0,10):
            if dataset_y_train[i][j]==1 and j!=0 and j!=1:
                dataset_y_train[i][j]=0
                dataset_y_train[i][2]=1
                



with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    # Create a session
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Build the graph
        cnn = C(patch_size=FLAGS.patch_size, num_filters_first_layer=num_filters_first_layer,
                  num_filters_second_layer=num_filters_second_layer, size_fully_connected_layer=size_fully_connected_layer,
                  lambda_reg = lambda_regs)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.cross_entropy)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        if FLAGS.run_name is None:
          timestamp = str(int(time.time()))
          out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        else:
          out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.run_name))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.cross_entropy)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        l2_summary = tf.summary.scalar("l2_regularizer", cnn.l2_loss)

        # Step time
        step_time_placeholder = tf.placeholder(dtype=tf.float32, shape={})
        step_time_summary = tf.summary.scalar("step_time", step_time_placeholder)
        last_step_time = 0.0

        # Since we have many channels, we will get filters only for one channel
        V = tf.slice(cnn.W_conv1, (0, 0, 0, 0), (-1, -1, -1, 1))
        # Bring into shape expected by image_summary
        V = tf.reshape(V, (-1, 5, 5, 1))
        image_summary_op = tf.summary.image("kernel_layer1", V, 5)
        # TODO: Add a summary for visualizing the filters from the second layer
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, image_summary_op, grad_summaries_merged, step_time_summary, l2_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. TF assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver()

        # Initialize all the variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
                A single training step
                """
            feed_dict = {
                cnn.x: x_batch,
                cnn.y_: y_batch,
                step_time_placeholder : last_step_time,
                cnn.keep_prob : keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.cross_entropy, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)
            train_summary_writer.flush()


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.x: x_batch,
              cnn.y_: y_batch,
              cnn.keep_prob : 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.cross_entropy, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)


        for i in range(FLAGS.num_epochs):
            batchx = dataset_x_train
            #batchx= tf.train.batch(dataset_x_train,chunks)
            batchy = dataset_y_train

            #batchy= tf.train.batch(dataset_y_train,chunks)
            batch=[batchx, batchy]
            begin = time.time()
            train_step(batch[0], batch[1])
            end = time.time()
            last_step_time = end - begin
            
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(mnist.validation.images, mnist.validation.labels, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
        a=mnist.test.labels
        #if the classification task is "easy", we need to change also the test set
        if easy_task==1:
            for i in range(0,len(a)):
                for j in range(0,10):
                    if a[i][j]==1 and j!=0 and j!=1:
                        a[i][j]=0
                        a[i][2]=1
        print("test accuracy %g"%cnn.accuracy.eval(feed_dict={
            cnn.x: mnist.test.images, cnn.y_: a, cnn.keep_prob: 1.0}))
        y=cnn.accuracy.eval(feed_dict={
            cnn.x: mnist.test.images, cnn.y_: a, cnn.keep_prob: 1.0})
        #write the results on a csv file
        with open('results.csv', 'a') as f:
            f.write(str(y))
            f.write("\n")

