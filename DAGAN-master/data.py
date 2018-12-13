import numpy as np
import pandas as pd
import os
from skimage import io
import math
np.random.seed(2591)

class Dataset(object):
    def __init__(self, csv_file, root_dir, transform=None, working_directory = '../working_directory/'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.info_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_size = (224, 224, 3)
        if transform == None:
            from torchvision import transforms as T
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(256),  # TODO set this value to 224, so the whole painting is cropped
                T.CenterCrop(224),
                ToNumpy()
            ])
        else: self.transform = transform
        self.working_directory = working_directory
        self.shape = (len(self),)+self.image_size

    def __len__(self):
        return len(self.info_file)


    def __getitem__(self, value):
        if (isinstance(value, int) or isinstance(value,np.int_)):
            return self.get_image(value)

        elif isinstance(value, tuple):
            if (isinstance(value[0],int) or isinstance(value[0], np.int_)):
                image = self.get_image(value[0])
                return image[value[1:]]
            else:
                image_array = self.get_image_array(value[0])
                return image_array[(slice(None, None, None),)+value[1:]]
        else:
            return self.get_image_array(value)

    def get_image(self,value):
        if (isinstance(value, int) or isinstance(value,np.int_)):
            img_name = os.path.join(self.root_dir,
                                    self.info_file.at[value, 'new_filename'])
            image = io.imread(img_name)
            if self.transform:
                image = self.transform(image)
            return image
        else:
            raise IndexError('Encountered unexpected index type in the method get_image, got {}'.format(type(value)))

    def get_image_array(self,value):
        if isinstance(value, slice):
            if value.start == None:
                start = 0
            else: start = value.start
            if value.stop == None:
                stop = len(self)
            else: stop = value.stop
            if value.step == None:
                step = 1
            else: step = value.step
            for idx in range(start, stop, step):
                img_name = os.path.join(self.root_dir,
                                        self.info_file.at[idx, 'new_filename'])
                image = io.imread(img_name)
                if self.transform:
                    image = self.transform(image)
                image = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))
                if idx == start:
                    image_array = image
                else:
                    image_array = np.concatenate((image_array, image),axis=0)
            return image_array

        elif isinstance(value, np.ndarray):
            if isinstance(value[0], np.int_):
                first = True
                for idx in value:
                    img_name = os.path.join(self.root_dir,
                                            self.info_file.at[idx, 'new_filename'])
                    image = io.imread(img_name)
                    if self.transform:
                        image = self.transform(image)
                    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
                    if first:
                        image_array = image
                        first = False
                    else:
                        image_array = np.concatenate((image_array, image), axis=0)
                return image_array

            elif isinstance(value[0], np.bool_):
                first = True
                idx = 0
                for boolean in value:
                    if boolean:
                        img_name = os.path.join(self.root_dir,
                                                self.info_file.at[idx, 'new_filename'])
                        image = io.imread(img_name)
                        if self.transform:
                            image = self.transform(image)
                        image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
                        if first:
                            image_array = image
                            first = False
                        else:
                            image_array = np.concatenate((image_array, image), axis=0)
                    idx += 1
                return image_array
            else:
                raise IndexError('Unrecognized indexing type encountered in numpy array, got {}'.format(type(value[0])))

        elif isinstance(value,list):
            if isinstance(value[0],int):
                first = True
                for idx in value:
                    img_name = os.path.join(self.root_dir,
                                            self.info_file.at[idx, 'new_filename'])
                    image = io.imread(img_name)
                    if self.transform:
                        image = self.transform(image)
                    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
                    if first:
                        image_array = image
                        first = False
                    else:
                        image_array = np.concatenate((image_array, image), axis=0)
                return image_array
            else:
                raise IndexError('Unreconized indexing type encountered in list, got {}'.format(type(value[0])))
        else:
            raise IndexError('Unrecognized slice indexing type encountered, got {}'.format(type(value)))

    def split_train_val(self, val_percentile, filename='temporary_file'):
        """
        Split dataset in a training set and validation set
        :param val_percentile: percentage of validation data
        :return: training dataset and validation dataset
        """

        filename_train = self.working_directory + filename + '_train.csv'
        filename_val = self.working_directory + filename + '_val.csv'
        idx_val = math.ceil((1-val_percentile)*len(self))
        info_file_train = self.info_file.loc[0:idx_val-1, :]
        info_file_val = self.info_file.loc[idx_val:, :]
        info_file_train.to_csv(filename_train, index=False)
        info_file_val.to_csv(filename_val, index=False)

        data_train = Dataset(csv_file=filename_train,root_dir=self.root_dir, transform=self.transform,
                             working_directory=self.working_directory)
        data_val = Dataset(csv_file=filename_val, root_dir=self.root_dir, transform=self.transform,
                           working_directory=self.working_directory)

        return data_train, data_val

    def split_train_val_test(self, val_percentile, test_percentile, filename='info_file'):
        """
        Split dataset in a training set and validation set
        :param val_percentile: percentage of validation data
        :return: training dataset and validation dataset
        """
        filename_train = self.working_directory+filename+'_train.csv'
        filename_val = self.working_directory+filename+'_val.csv'
        filename_test = self.working_directory+filename+'_test.csv'
        idx_val = math.ceil((1-val_percentile-test_percentile)*len(self))
        idx_test = math.ceil((1-test_percentile)*len(self))
        info_file_train = self.info_file.loc[0:idx_val-1, :]
        info_file_val = self.info_file.loc[idx_val:idx_test-1, :]
        info_file_test = self.info_file.loc[idx_test:, :]
        info_file_train.to_csv(filename_train, index=False)
        info_file_val.to_csv(filename_val, index=False)
        info_file_test.to_csv(filename_test, index=False)

        data_train = Dataset(csv_file=filename_train, root_dir=self.root_dir, transform=self.transform,
                             working_directory=self.working_directory)
        data_val = Dataset(csv_file=filename_val, root_dir=self.root_dir,transform=self.transform,
                           working_directory=self.working_directory)
        data_test = Dataset(csv_file=filename_test, root_dir=self.root_dir,transform=self.transform,
                            working_directory=self.working_directory)

        return data_train, data_val, data_test

    def to_multiclass(self):
        filename = self.working_directory + 'temporary_file.csv'
        self.info_file.to_csv(filename, index=False)
        return MultiClassDataset(csv_file=filename, root_dir=self.root_dir, transform=self.transform,
                                 working_directory=self.working_directory)

class OneClassDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, working_directory = '../working_directory/'):
        super().__init__(csv_file=csv_file, root_dir=root_dir, transform=transform, working_directory= working_directory)
        self.assert_one_class()

    def assert_one_class(self):
        df = self.info_file
        classes = list(df['style'].unique())
        if len(classes)>1:
            raise ValueError('More than 1 class present in a OneClassDataset object')


class MultiClassDataset(object):
    def __init__(self, csv_file, root_dir, transform=None, working_directory = '../working_directory/'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.info_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.classes, self.class_to_idx, self.idx_to_class = self.find_classes()
        self.info_file_classes = self.split_info_file_in_classes()
        self.nb_classes = len(self.classes)
        self.class_lengths = [len(self.info_file_classes[i]) for i in range(self.nb_classes)]
        self.image_size = (224,224,3)
        if transform == None:
            from torchvision import transforms as T
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(256),  # TODO set this value to 224, so the whole painting is cropped
                T.CenterCrop(224),
                ToNumpy()
            ])
        else: self.transform = transform
        self.working_directory = working_directory
        self.shape = (self.nb_classes, np.min(np.array(self.class_lengths))) + self.image_size


    def __len__(self):
        return len(self.info_file)


    def __getitem__(self, value):
        if isinstance(value,int):
            csv_filename = self.working_directory + 'DAGAN_temporaryfile.csv'
            self.info_file_classes[value].to_csv(csv_filename, index=False)
            return OneClassDataset(csv_file=csv_filename,root_dir = self.root_dir, transform = self.transform,
                                   working_directory=self.working_directory)
        if isinstance(value, tuple):
            if not (isinstance(value[0], int) or isinstance(value[0],np.int_)):
                raise IndexError('first index of Multiclass object should be integer')
            csv_filename = self.working_directory + 'DAGAN_temporaryfile.csv'
            self.info_file_classes[value[0]].to_csv(csv_filename, index=False)
            one_class_data = OneClassDataset(csv_file=csv_filename,root_dir = self.root_dir, transform = self.transform,
                                             working_directory=self.working_directory)
            return one_class_data[value[1:]]
        else:
            return self.get_slice(value)


    def get_slice(self,value):
        if isinstance(value, slice):
            if value.start == None:
                start = 0
            else: start = value.start
            if value.stop == None:
                stop = len(self)
            else: stop = value.stop
            if value.step == None:
                step = 1
            else: step = value.step
            iterator = range(start, stop, step)
        elif (isinstance(value[0],bool) or isinstance(value[0], np.bool_)):
            idx = 0
            iterator = []
            for boolean in value:
                if boolean:
                    iterator.append(idx)
                idx += 1
        else:
            iterator = value

        csv_filename = self.working_directory + 'DAGAN_temporaryfile_class.csv'
        first = True
        for label_idx in iterator:
            if first:
                csv_file = self.info_file_classes[label_idx]
                first = False
            else:
                csv_file = pd.concat([csv_file, self.info_file_classes[label_idx]])
        csv_file.to_csv(csv_filename, index=False)
        return MultiClassDataset(csv_file=csv_filename, root_dir=self.root_dir, transform=self.transform,
                                 working_directory=self.working_directory)

    def find_classes(self):
        df = self.info_file
        classes = list(df['style'].unique())
        classes.sort()
        class_to_idx = {val: idx for (idx, val) in enumerate(classes)}
        idx_to_class = {idx: val for (idx, val) in enumerate(classes)}
        return classes, class_to_idx, idx_to_class

    def split_info_file_in_classes(self):
        info_files_classes = []
        for label in self.classes:
            info_files_classes.append(self.info_file.loc[self.info_file['style'] == label])
        return info_files_classes

    def flatten_to_dataset(self):
        filename = self.working_directory + 'temporary_file.csv'
        self.info_file.to_csv(filename, index=False)
        return Dataset(csv_file=filename, root_dir=self.root_dir, transform=self.transform,
                                 working_directory=self.working_directory)



class CustomDAGANDataset(object):
    pass



class DAGANDataset(object):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 gen_labels = None):
        """
        :param batch_size: The batch size to use for the data loader
        :param last_training_class_index: The final index for the training set, used to restrict the training set
        if needed. E.g. if training set is 1200 classes and last_training_class_index=900 then only the first 900
        classes will be used
        :param reverse_channels: A boolean indicating whether we need to reverse the colour channels e.g. RGB to BGR
        :param num_of_gpus: Number of gpus to use for training
        :param gen_batches: How many batches to use from the validation set for the end of epoch generations
        """
        self.x_train, self.x_test, self.x_val = self.load_dataset(last_training_class_index)
        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.reverse_channels = reverse_channels
        self.test_samples_per_label = gen_batches
        if gen_labels:
            self.choose_gen_labels = gen_labels
            if isinstance(gen_labels, int):
                gen_labels = [gen_labels]
            x_gen = self.x_val[gen_labels]
            x_gen = x_gen.flatten_to_dataset()
        else:
            x_gen = self.x_val.flatten_to_dataset()
            self.choose_gen_labels = self.x_val.classes
        self.choose_gen_samples = np.random.choice(len(x_gen), self.test_samples_per_label, replace=True)
        self.x_gen = x_gen[self.choose_gen_samples]
        self.gen_batches = gen_batches

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.indexes = {"train": 0, "val": 0, "test": 0, "gen": 0}
        self.datasets = {"train": self.x_train, "gen": self.x_gen,
                         "val": self.x_val,
                         "test": self.x_test}

        self.image_height = self.x_train.shape[2]
        self.image_width = self.x_train.shape[3]
        self.image_channel = self.x_train.shape[4]
        self.training_data_size = self.x_train.shape[0] * self.x_train.shape[1]
        self.validation_data_size = gen_batches * self.batch_size
        self.testing_data_size = self.x_test.shape[0] * self.x_test.shape[1]
        self.generation_data_size = self.gen_batches * self.batch_size

    def load_dataset(self, last_training_class_index):
        """
        Loads the dataset into the data loader class. To be implemented in all classes that inherit
        DAGANImbalancedDataset
        :param last_training_class_index: last_training_class_index: The final index for the training set,
        used to restrict the training set if needed. E.g. if training set is 1200 classes and
        last_training_class_index=900 then only the first 900 classes will be used
        """
        raise NotImplementedError

    def preprocess_data(self, x):
        """
        Preprocesses data such that their values lie in the -1.0 to 1.0 range so that the tanh activation gen output
        can work properly
        :param x: A data batch to preprocess
        :return: A preprocessed data batch
        """
        if not isinstance(x,np.ndarray):
            raise TypeError('Only numpy arrays can be used with the method preprocess_data, now got {}'.format(type(x)))
        x = 2 * x - 1
        if self.reverse_channels:
            reverse_photos = np.ones(shape=x.shape)
            for channel in range(x.shape[-1]):
                reverse_photos[:, :, :, x.shape[-1] - 1 - channel] = x[:, :, :, channel]
            x = reverse_photos
        return x

    def reconstruct_original(self, x):
        """
        Applies the reverse operations that preprocess_data() applies such that the data returns to their original form
        :param x: A batch of data to reconstruct
        :return: A reconstructed batch of data
        """
        if not isinstance(x,np.ndarray):
            raise TypeError('Only numpy arrays can be used with the method reconstruct original, now got {}'.format(type(x)))
        x = (x + 1) / 2
        return x

    def shuffle(self, x):
        """
        Shuffles the data batch along it's first axis
        :param x: A data batch
        :return: A shuffled data batch
        """
        raise NotImplementedError

        # indices = np.arange(len(x))
        # np.random.shuffle(indices)
        # x = x[indices]
        # return x

    def get_batch(self, dataset_name):
        """
        Generates a data batch to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc
        :return: A data batch
        """

        x_input_batch_a = []
        x_input_batch_b = []

        for i in range(self.batch_size):
            class_label = np.random.choice(self.datasets[dataset_name].nb_classes)
            dataset_class = self.datasets[dataset_name][class_label]
            sample1 = np.random.choice(len(dataset_class))
            sample2 = np.random.choice(len(dataset_class))
            x_input_batch_a.append(dataset_class[sample1])
            x_input_batch_b.append(dataset_class[sample2])

        x_input_batch_a = np.array(x_input_batch_a)
        x_input_batch_b = np.array(x_input_batch_b)

        return self.preprocess_data(x_input_batch_a), self.preprocess_data(x_input_batch_b)

    def get_next_gen_batch(self):
        """
        Provides a batch that contains data to be used for generation
        :return: A data batch to use for generation
        """
        if self.indexes["gen"] >= self.batch_size * self.gen_batches:
            self.indexes["gen"] = 0
        x_input_batch_a = self.datasets["gen"][self.indexes["gen"]:self.indexes["gen"]+self.batch_size]
        self.indexes["gen"] += self.batch_size
        return self.preprocess_data(x_input_batch_a)

    def get_multi_batch(self, dataset_name):
        """
        Returns a batch to be used for training or evaluation for multi gpu training
        :param set_name: The name of the data-set to use e.g. "train", "test" etc
        :return: Two batches (i.e. x_i and x_j) of size [num_gpus, batch_size, im_height, im_width, im_channels). If
        the set is "gen" then we only return a single batch (i.e. x_i)
        """
        x_input_a_batch = []
        x_input_b_batch = []
        if dataset_name == "gen":
            x_input_a = self.get_next_gen_batch()
            for n_batch in range(self.num_of_gpus):
                x_input_a_batch.append(x_input_a)
            x_input_a_batch = np.array(x_input_a_batch)
            return x_input_a_batch
        else:
            for n_batch in range(self.num_of_gpus):
                x_input_a, x_input_b = self.get_batch(dataset_name)
                x_input_a_batch.append(x_input_a)
                x_input_b_batch.append(x_input_b)

            x_input_a_batch = np.array(x_input_a_batch)
            x_input_b_batch = np.array(x_input_b_batch)

            return x_input_a_batch, x_input_b_batch

    def get_train_batch(self):
        """
        Provides a training batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for training
        """
        x_input_a, x_input_b = self.get_multi_batch("train")
        return x_input_a, x_input_b

    def get_test_batch(self):
        """
        Provides a test batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for evaluation
        """
        x_input_a, x_input_b = self.get_multi_batch("test")
        return x_input_a, x_input_b

    def get_val_batch(self):
        """
        Provides a val batch
        :return: Returns a tuple of two data batches (i.e. x_i and x_j) to be used for evaluation
        """
        x_input_a, x_input_b = self.get_multi_batch("val")
        return x_input_a, x_input_b

    def get_gen_batch(self):
        """
        Provides a gen batch
        :return: Returns a single data batch (i.e. x_i) to be used for generation on unseen data
        """
        x_input_a = self.get_multi_batch("gen")
        return x_input_a

class DAGANImbalancedDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        """
                :param batch_size: The batch size to use for the data loader
                :param last_training_class_index: The final index for the training set, used to restrict the training set
                if needed. E.g. if training set is 1200 classes and last_training_class_index=900 then only the first 900
                classes will be used
                :param reverse_channels: A boolean indicating whether we need to reverse the colour channels e.g. RGB to BGR
                :param num_of_gpus: Number of gpus to use for training
                :param gen_batches: How many batches to use from the validation set for the end of epoch generations
                """
        self.x_train, self.x_test, self.x_val = self.load_dataset(last_training_class_index)

        self.training_data_size = np.sum([len(self.x_train[i]) for i in range(self.x_train.shape[0])])
        self.validation_data_size = np.sum([len(self.x_val[i]) for i in range(self.x_val.shape[0])])
        self.testing_data_size = np.sum([len(self.x_test[i]) for i in range(self.x_test.shape[0])])
        self.generation_data_size = gen_batches * batch_size

        self.num_of_gpus = num_of_gpus
        self.batch_size = batch_size
        self.reverse_channels = reverse_channels

        val_dict = dict()
        idx = 0
        for i in range(self.x_val.shape[0]):
            temp = self.x_val[i]
            for j in range(len(temp)):
                val_dict[idx] = {"sample_idx": j, "label_idx": i}
                idx += 1
        choose_gen_samples = np.random.choice([i for i in range(self.validation_data_size)],
                                                   size=self.generation_data_size)


        self.x_gen = np.array([self.x_val[val_dict[idx]["label_idx"]][val_dict[idx]["sample_idx"]]
                               for idx in choose_gen_samples])

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0

        self.indexes = {"train": 0, "val": 0, "test": 0, "gen": 0}
        self.datasets = {"train": self.x_train, "gen": self.x_gen,
                         "val": self.x_val,
                         "test": self.x_test}

        self.gen_data_size = gen_batches * self.batch_size
        self.image_height = self.x_train[0][0].shape[0]
        self.image_width = self.x_train[0][0].shape[1]
        self.image_channel = self.x_train[0][0].shape[2]

    def get_batch(self, set_name):
        """
        Generates a data batch to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc
        :return: A data batch
        """
        choose_classes = np.random.choice(len(self.datasets[set_name]), size=self.batch_size)

        x_input_batch_a = []
        x_input_batch_b = []

        for i in range(self.batch_size):
            choose_samples = np.random.choice(len(self.datasets[set_name][choose_classes[i]]),
                                              size=2 * self.batch_size,
                                              replace=True)

            choose_samples_a = choose_samples[:self.batch_size]
            choose_samples_b = choose_samples[self.batch_size:]
            current_class_samples = self.datasets[set_name][choose_classes[i]]
            x_input_batch_a.append(current_class_samples[choose_samples_a[i]])
            x_input_batch_b.append(current_class_samples[choose_samples_b[i]])

        x_input_batch_a = np.array(x_input_batch_a)
        x_input_batch_b = np.array(x_input_batch_b)

        return self.preprocess_data(x_input_batch_a), self.preprocess_data(x_input_batch_b)

    def get_next_gen_batch(self):
        """
        Provides a batch that contains data to be used for generation
        :return: A data batch to use for generation
        """
        if self.indexes["gen"] >= self.gen_data_size:
            self.indexes["gen"] = 0
        x_input_batch_a = self.datasets["gen"][self.indexes["gen"]:self.indexes["gen"]+self.batch_size]
        self.indexes["gen"] += self.batch_size
        return self.preprocess_data(x_input_batch_a)

    def get_multi_batch(self, set_name):
        """
        Returns a batch to be used for training or evaluation for multi gpu training
        :param set_name: The name of the data-set to use e.g. "train", "test" etc
        :return: Two batches (i.e. x_i and x_j) of size [num_gpus, batch_size, im_height, im_width, im_channels). If
        the set is "gen" then we only return a single batch (i.e. x_i)
        """
        x_input_a_batch = []
        x_input_b_batch = []
        if set_name == "gen":
            x_input_a = self.get_next_gen_batch()
            for n_batch in range(self.num_of_gpus):
                x_input_a_batch.append(x_input_a)
            x_input_a_batch = np.array(x_input_a_batch)
            return x_input_a_batch
        else:
            for n_batch in range(self.num_of_gpus):
                x_input_a, x_input_b = self.get_batch(set_name)
                x_input_a_batch.append(x_input_a)
                x_input_b_batch.append(x_input_b)

            x_input_a_batch = np.array(x_input_a_batch)
            x_input_b_batch = np.array(x_input_b_batch)

            return x_input_a_batch, x_input_b_batch


class OmniglotDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 gen_labels=None):
        super(OmniglotDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                   gen_batches,gen_labels=gen_labels)

    def load_dataset(self, gan_training_index):
        self.x = np.load("datasets/omniglot_data.npy")
        self.x = self.x / np.max(self.x)
        x_train, x_test, x_val = self.x[:1200], self.x[1200:1600], self.x[1600:]
        x_train = x_train[:gan_training_index]
        return x_train, x_test, x_val

class OmniglotImbalancedDAGANDataset(DAGANImbalancedDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        super(OmniglotImbalancedDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels,
                                                             num_of_gpus, gen_batches)
    def load_dataset(self, last_training_class_index):
        x = np.load("datasets/omniglot_data.npy")
        x_temp = []
        for i in range(x.shape[0]):
            choose_samples = np.random.choice([i for i in range(1, 15)])
            x_temp.append(x[i, :choose_samples])
        self.x = np.array(x_temp)
        self.x = self.x / np.max(self.x)
        x_train, x_test, x_val = self.x[:1200], self.x[1200:1600], self.x[1600:]
        x_train = x_train[:last_training_class_index]

        return x_train, x_test, x_val


class VGGFaceDAGANDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches):
        super(VGGFaceDAGANDataset, self).__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                  gen_batches)

    def load_dataset(self, gan_training_index):

        self.x = np.load("datasets/vgg_face_data.npy")
        self.x = self.x / np.max(self.x)
        self.x = np.reshape(self.x, newshape=(2354, 100, 64, 64, 3))
        x_train, x_test, x_val = self.x[:1803], self.x[1803:2300], self.x[2300:]
        x_train = x_train[:gan_training_index]

        return x_train, x_test, x_val

class PaintingsDataset(DAGANDataset):
    def __init__(self, batch_size, last_training_class_index, reverse_channels, num_of_gpus, gen_batches,
                 gen_labels=None, csv_file='../data_info_files/final_train_info.csv',
                 root_dir='../../DeepLearningData/train_reduced', transform = None,
                 working_directory='../working_directory/'):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.working_directory = working_directory
        self.transform = transform
        super().__init__(batch_size, last_training_class_index, reverse_channels, num_of_gpus,
                                                   gen_batches, gen_labels=gen_labels)



    def load_dataset(self, gan_training_index):
        dataset = Dataset(csv_file=self.csv_file,
                          root_dir=self.root_dir,
                          transform=None,
                          working_directory=self.working_directory)
        x_train, x_val, x_test = dataset.split_train_val_test(0.15, 0.15)
        x_train = x_train.to_multiclass()
        x_val = x_val.to_multiclass()
        x_test = x_test.to_multiclass()
        return x_train, x_test, x_val


class ToNumpy(object):
    """Transform PIL image to numpy array
    """

    def __call__(self, pic):
        return np.array(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

if __name__ == '__main__':

    from torchvision import transforms as T
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),  # TODO set this value to 224, so the whole painting is cropped
        T.CenterCrop(224),
        ToNumpy()
    ])
    data = Dataset('../data_info_files/final_train_info.csv', '../../DeepLearningData/train_reduced',
                                         transform=transform)
    multi_class_data = data.to_multiclass()
    test = multi_class_data[0,3:8]
