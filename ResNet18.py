import random
seed = 231
random.seed(seed) # fix the datasets

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import TensorDataset
from Dataset import PaintingDataset
import os
import torchvision
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as T

# import torchnet as tnt
# from torchnet.meter import ConfusionMeter

import numpy as np

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
img_folder = os.path.join(os.path.curdir, 'train_reduced')
data_csv_file = os.path.join(os.path.curdir, 'final_train_info.csv')
num_workers = 1

filter_subset = False # True if we want to filter to just train _1
balanced_dset = True # True if I want equal # of paintings per artist, false if I want to use all available per artist

## HELPER FUNCTIONS

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def train(model, loss_fn, optimizer, loader_train, loader_val, train_acc, val_acc, num_epochs=1):
    train_loss_hist = []
    # val_loss_hist = []
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())
            scores = model(x_var)
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # record training loss history
        train_loss_hist.append(loss)

        # record training and validation accuracy at the end of each epoch
        train_acc.append(check_accuracy(model, loader_train))
        val_acc.append(check_accuracy(model, loader_val))

    return [train_acc, val_acc, train_loss_hist]


def check_accuracy(model, loader):
    print('Checking accuracy!')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        y = y.view(-1, 1).type(ytype)
        x_var = Variable(x.type(dtype), volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return 100 * acc


def confusion_matrix(model, loader, conf):
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        y = y.view(-1, 1).type(ytype)
        x_var = Variable(x.type(dtype), volatile=True)
        scores = model(x_var)

        conf.add(scores.data, y)




## THIS VERSION OF SCRIPT HAS EQUAL NUMBER OF PAINTINGS PER ARTIST
num_train = 240
num_val = 30
num_test = 30
num_samples = num_train + num_val + num_test # threshold to include an artist
b_size = 32 # batch size for the data loaders

mean_resnet = np.array([0.485, 0.456, 0.406]) # This I found from internet (mean values for ImageNet, we can check if this is correct)
std_resnet = np.array([0.229, 0.224, 0.225])

train_transform = T.Compose([
    T.RandomResizedCrop(224),
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

dset = PaintingDataset(root_dir=img_folder,csv_file=data_csv_file,transform=None)
train_dset, val_dset, test_dset = dset.split_train_val_test(0.15,0.15)

train_dset.transform = train_transform
val_dset.transform = val_transform
test_dset.transform = val_transform

loader_train = DataLoader(train_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
loader_val = DataLoader(val_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)
loader_test = DataLoader(test_dset, batch_size=b_size, shuffle=True, num_workers=num_workers)

nb_classes = len(dset.classes)

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

train_acc, val_acc, train_loss = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, train_acc, val_acc, num_epochs = 5)



# now we allow all of the network to change, but by less
for param in model_conv.parameters():
    param.requires_grad = True

optimizer_conv = optim.Adam(model_conv.parameters(), lr=1e-10)

train_acc, val_acc, train_loss = train(model_conv, loss_fn, optimizer_conv, loader_train, loader_val, train_acc, val_acc, num_epochs = 1)

