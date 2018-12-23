from data import PaintingsDataset

data = PaintingsDataset(batch_size=32, last_training_class_index=900, reverse_channels=True,
                                    num_of_gpus=1, gen_batches=10, csv_file=args.infofile, root_dir = args.data_dir)
#init experiment