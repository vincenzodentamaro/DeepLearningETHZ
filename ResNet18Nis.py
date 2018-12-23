import random
import math
seed = 231
random.seed(seed) # fix the datasets
from helper_functions import train, check_accuracy, confusion_matrix, reset, Flatten, ImplementationError, write_results

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from painting_loader import PaintingFolder

import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as T



import numpy as np
import timeit


dtype = torch.FloatTensor
ytype = torch.LongTensor
ytype_cuda = torch.cuda.LongTensor
if (torch.cuda.is_available()):
   dtype = torch.cuda.FloatTensor
print(ytype)
print(dtype)
print_every = 100


dat_folder = 'working_directory_andreas/'
img_folder = 'reduced/'
num_workers = 4

filter_subset = False # True if we want to filter to just train _1
balanced_dset = True # True if I want equal # of paintings per artist, false if I want to use all available per artist

## THIS VERSION OF SCRIPT HAS EQUAL NUMBER OF PAINTINGS PER ARTIST
num_train = 4000
num_val = 500
num_test = 500
num_samples = num_train + num_val + num_test # threshold to include an artist
b_size = 50 # batch size for the data loaders



import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import os

t = pd.read_csv(dat_folder + 'all_data_info.csv')

# filter down (if needed)
if (filter_subset):
    t = t[t['new_filename'].str.startswith('1')]
    t = t[t['in_train']]

t.head()
# print(t.shape)

x = list(t['style'].value_counts())
# list of all artists to include
temp = t['style'].value_counts()
threshold = num_samples
# threshold = 500
artists = temp[temp >= threshold].index.tolist()
num_artists = len(artists)

print(str(len(artists)) + ' styles being classified')

# pull train and val data for just those artists
train_dfs = []
val_dfs = []
test_dfs = []

for a in artists:
    df = t[t['style'].str.startswith(a, na=False)].sample(n=num_samples, random_state=seed)
    t_df = df.sample(n=num_train, random_state=seed)
    rest_df = df.loc[~df.index.isin(t_df.index)]
    v_df = rest_df.sample(n=num_val, random_state=seed)
    te_df = rest_df.loc[~rest_df.index.isin(v_df.index)]
    
    train_dfs.append(t_df)
    val_dfs.append(v_df)
    test_dfs.append(te_df)

train_df = pd.concat(train_dfs)
val_df = pd.concat(val_dfs)
test_df = pd.concat(test_dfs)

print(train_df.shape)
print(val_df.shape)
print(test_df.shape)
print("Done")


mean_resnet = np.array([0.485, 0.456, 0.406])
std_resnet = np.array([0.229, 0.224, 0.225])
        
train_transform = T.Compose([
        T.RandomSizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean_resnet, std_resnet)
    ])
val_transform = T.Compose([
        T.Scale(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean_resnet, std_resnet)
    ])

train_dset = PaintingFolder(img_folder, train_transform, train_df)
loader_train = DataLoader(train_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
    
val_dset = PaintingFolder(img_folder, val_transform, val_df)
loader_val = DataLoader(val_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)

test_dset = PaintingFolder(img_folder, val_transform, test_df)
loader_test = DataLoader(test_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)

print("Done")



import torchvision 

# transfer learning on top of ResNet (only replacing final FC layer)
# model_conv = torchvision.models.resnet18(pretrained=True)

model_conv = models.resnet18(pretrained=True)

for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_artists)

if torch.cuda.is_available():
    model_conv = model_conv.cuda()

loss_fn = nn.CrossEntropyLoss().type(dtype)

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=1e-3)


train_acc = []
val_acc = []

start_time = timeit.default_timer()
train_acc, val_acc, train_loss = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, train_acc, val_acc, num_epochs = 5)

print()
print(str(timeit.default_timer() - start_time) + " seconds taken")

# now we allow all of the network to change, but by less
for param in model_conv.parameters():
    param.requires_grad = True

optimizer_conv = optim.Adam(model_conv.parameters(), lr=1e-4, weight_decay=1e-2)

start_time = timeit.default_timer()
train_acc, val_acc, train_loss = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, train_acc, val_acc, num_epochs = 3)

print()
print(str(timeit.default_timer() - start_time) + " seconds taken")


# check_accuracy(model_conv, loader_train)
# check_accuracy(model_conv, loader_val)
check_accuracy(model_conv, loader_test)


check_accuracy_topX(model_conv, loader_train, top=3)
check_accuracy_topX(model_conv, loader_val, top=3)
check_accuracy_topX(model_conv, loader_test, top=3)



epochs = np.arange(len(train_acc)) + 1
print(train_acc)
print(val_acc)
print(epochs)

# Plot the points using matplotlib
plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)
plt.xlabel('Epoch')
plt.ylabel('Top-1 Classification Accuracy')
plt.title('ResNet-18 with Transfer Learning')
plt.legend(['Training', 'Validation'])

# y ticks
plt.gca().set_ylim(ymin=0, ymax=100)
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in vals])

# x ticks
# vals = plt.gca().get_xticks()
# plt.gca().set_xticklabels(['{:0.0f}'.format(x) for x in vals])

plt.show()
