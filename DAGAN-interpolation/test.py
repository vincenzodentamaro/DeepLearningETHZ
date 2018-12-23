from data import PaintingsDataset

data = PaintingsDataset(batch_size=35, last_training_class_index=900, reverse_channels=True,
                                    num_of_gpus=num_gpus, gen_batches=10, csv_file=args.infofile, root_dir = args.data_dir)
#init experiment