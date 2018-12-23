import torch
import torch.nn as nn
from torch.autograd import Variable
import timeit
import pandas as pd


def main():
    dtype = torch.FloatTensor
    ytype = torch.LongTensor
    ytype_cuda = torch.cuda.LongTensor
    if (torch.cuda.is_available()):
        dtype = torch.cuda.FloatTensor
    print(ytype)
    print(dtype)
    print_every = 100

if __name__ == '__main__':
    main()

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def train(model, loss_fn, optimizer, loader_train, loader_val,train_acc, val_acc, num_epochs=1):

    train_loss_hist = []
    train_time = []
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        start_time = timeit.default_timer()
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
        # record epoch time
        train_time.append(timeit.default_timer() - start_time) #TODO remove print statement when timing results
        # record training loss history
        train_loss_hist.append(loss.item())

        # record training and validation accuracy at the end of each epoch
        train_acc.append(check_accuracy(model, loader_train))
        val_acc.append(check_accuracy(model, loader_val))

    return [train_acc, val_acc, train_loss_hist, train_time]


def check_accuracy(model, loader):
    dtype = torch.FloatTensor
    ytype = torch.LongTensor
    ytype_cuda = torch.cuda.LongTensor
    if (torch.cuda.is_available()):
        dtype = torch.cuda.FloatTensor
    print('Checking accuracy!')
    num_correct = 0
    num_samples = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(dtype))#, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return 100 * acc


def confusion_matrix(model, loader, conf):
    dtype = torch.FloatTensor
    ytype = torch.LongTensor
    ytype_cuda = torch.cuda.LongTensor
    if (torch.cuda.is_available()):
        dtype = torch.cuda.FloatTensor
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        y = y.view(-1, 1).type(ytype)
        x_var = Variable(x.type(dtype), volatile=True)
        scores = model(x_var)

        conf.add(scores.data, y)

class ImplementationError(Exception):
    pass

def write_results(results):
    column_header = ['epoch', 'accuracy_train', 'accuracy_val', 'training_loss', 'runtime']
    rows = []
    for idx in range(len(results[0])):
        row = [idx, results[0][idx], results[1][idx], results[2][idx], results[3][idx]]
        rows.append(row)

    return pd.DataFrame(rows,columns=column_header)


