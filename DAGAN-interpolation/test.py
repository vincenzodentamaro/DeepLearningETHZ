from data import PaintingsDataset

data = PaintingsDataset(batch_size=8, last_training_class_index=900, reverse_channels=True,
                        num_of_gpus=1, gen_batches=8, csv_file='../data_info_files/info_file_testrun_gen.csv',
                        root_dir = '../../DeepLearningData/train_reduced')
#init experiment