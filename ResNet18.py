import random
seed = 231
random.seed(seed) # fix the datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Dataset import PaintingDataset, AugmentedPaintingDataset
import os
import torchvision
import torchvision.transforms as T
import timeit
from helper_functions import train, check_accuracy, confusion_matrix, reset, Flatten, ImplementationError, write_results
import numpy as np
import argparse
import pandas as pd

if __name__ == '__main__':
    ## HANDLE COMMAND LINE INPUT ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_augmentation", help='Data augmentation to be used, either "none", "standard" or "extra"',
                        type=str, default='standard')
    parser.add_argument("--resultfile", help='Name of the resultfile that will be created',
                        type=str, default='result_ResNet18.csv')
    parser.add_argument("--datafolder", help='name of the folder with the data', default='train_reduced')
    parser.add_argument("--datafolder_extra", help='name of the folder with the extra generated data', default='generated_data')
    parser.add_argument("--infofile_extra", help='name of the csv file with the sample information of the extra '
                                                 'generated data', default='info_file_gen.csv')
    parser.add_argument("--infofile", help= 'name of the csv file with the sample information', default='final_train_info.csv')
    args = parser.parse_args()

    if args.resultfile[-4:] != '.csv':
        raise ValueError('Expecting a csv file as result filename, got {} as extension'.format(args.resultfile[-4:]))

    resultfilename1 = './result_files/' + args.resultfile[:-4]+'1'+args.resultfile[-4:]
    resultfilename2 = './result_files/' + args.resultfile[:-4]+'2'+args.resultfile[-4:]

    ## For running on GPU

    dtype = torch.FloatTensor
    ytype = torch.LongTensor
    ytype_cuda = torch.cuda.LongTensor
    if (torch.cuda.is_available()):
        dtype = torch.cuda.FloatTensor
    print(ytype)
    print(dtype)
    print_every = 100


    ## ALL MAIN ARGUMENTS FOR THE SCRIPT ##

    # dat_folder = os.path.join(os.path.curdir,'train_reduced')

    # img_folder = os.path.join(os.path.curdir, '../DeepLearningData/train_reduced')
    # data_csv_file = os.path.join(os.path.curdir, 'info_file_testrun.csv')
    # img_folder_extra = os.path.join(os.path.curdir, '../DeepLearningData/train_reduced')
    # data_csv_file_extra = os.path.join(os.path.curdir, 'info_file_testrun_gen.csv') #TODO comment and uncomment right stuff
    img_folder_extra = os.path.join(os.path.curdir, args.datafolder_extra)
    data_csv_file_extra = os.path.join(os.path.curdir, 'data_info_files', args.infofile_extra)
    img_folder = os.path.join(os.path.curdir, args.datafolder)
    data_csv_file = os.path.join(os.path.curdir, 'data_info_files', args.infofile)
    num_workers = 1



    ## INITIALIZE DATASETS

    b_size = 4 #TODO: set back to 32

    mean_resnet = np.array([0.485, 0.456, 0.406]) # This I found from internet (mean values for ImageNet, we can check if this is correct)
    std_resnet = np.array([0.229, 0.224, 0.225])

    val_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),  # TODO set this value to 224, so the whole painting is cropped
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean_resnet, std_resnet)
    ])

    if args.data_augmentation == 'extra':
        train_transform = T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean_resnet, std_resnet)
        ])

        total_dset = PaintingDataset(root_dir=img_folder, csv_file=data_csv_file, transform=None)
        extra_dset = PaintingDataset(root_dir = img_folder_extra, csv_file=data_csv_file_extra, transform=None)
        augmented_dset = AugmentedPaintingDataset(total_dset, extra_dset)
        train_dset, val_dset, test_dset = augmented_dset.split_train_val_test(0.15, 0.15)

        train_dset.main_dataset.transform = train_transform
        train_dset.extra_dataset.transform = train_transform
        val_dset.main_dataset.transform = val_transform
        val_dset.extra_dataset.transform = val_transform
        test_dset.main_dataset.transform = val_transform
        test_dset.extra_dataset.transform = val_transform

        loader_train = DataLoader(train_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        loader_val = DataLoader(val_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        loader_test = DataLoader(test_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)

        nb_classes = len(total_dset.classes)
    else:
        if args.data_augmentation == 'standard':
            train_transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean_resnet, std_resnet)
            ])
        elif args.data_augmentation == 'none':
            train_transform = T.Compose([
                T.ToPILImage(),
                T.Resize(256),  # TODO set this value to 224, so the whole painting is cropped
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean_resnet, std_resnet)
            ])
        else:
            raise ValueError('data augmentation argument is not recognized, use either "none", "standard" or "GAN"')

        total_dset = PaintingDataset(root_dir=img_folder,csv_file=data_csv_file,transform=None)
        train_dset, val_dset, test_dset = total_dset.split_train_val_test(0.15,0.15)

        train_dset.transform = train_transform
        val_dset.transform = val_transform
        test_dset.transform = val_transform

        loader_train = DataLoader(train_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        loader_val = DataLoader(val_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
        loader_test = DataLoader(test_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)

        nb_classes = len(total_dset.classes)

    ## IMPLEMENT AND TRAIN RESNET18

    # transfer learning on top of ResNet (only replacing final FC layer)
    # model_conv = torchvision.models.resnet18(pretrained=True)
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, nb_classes)

    if torch.cuda.is_available():
        model_conv = model_conv.cuda()

    loss_fn = nn.CrossEntropyLoss().type(dtype)

    # Observe that only parameters of final layer are being optimized as
    # opoosed to before.
    optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=1e-3)

    train_acc = []
    val_acc = []

    results = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, num_epochs = 5)
    resultfile1 = write_results(results)
    resultfile1.to_csv(os.path.join(os.path.curdir, resultfilename1))
    test_accuracy = check_accuracy(model_conv,loader_test)
    print('Test accuracy after training last layer: {}'.format(test_accuracy))




    # now we allow all of the network to change, but by less
    for param in model_conv.parameters():
        param.requires_grad = True

    optimizer_conv = optim.Adam(model_conv.parameters(), lr=1e-10)

    results = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, num_epochs = 1)
    resultfile2 = write_results(results)
    resultfile2.to_csv(os.path.join(os.path.curdir, resultfilename2))
    test_accuracy = check_accuracy(model_conv,loader_test)
    print('Test accuracy after training total network with small learning rate: {}'.format(test_accuracy))


