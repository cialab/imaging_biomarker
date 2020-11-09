from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

from base_dataloader import HDF5Dataset
from base_slideModel import Attention

from torch.cuda import memory_cached
from torch.cuda import empty_cache

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--patch_size', type=int, metavar='PS',
                    help='patch size')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=0.0005, metavar='R',
                    help='weight decay')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--split', type=float, default=0.8, metavar='Sp',
                    help='random seed (default: 1)')
parser.add_argument('--file_path',
                    help='where to search for dataset .h5')
parser.add_argument('--recursive', type=bool, default=True,
                    help='to search recursively in file_path')
parser.add_argument('--load_data', type=bool, default=False,
                    help='load all data first?')
parser.add_argument('--transform', type=bool, default=True,
                    help='map values?')
parser.add_argument('--data_cache_size', type=int, default=3,
                    help='not sure')                    
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
                    

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(HDF5Dataset(file_path=args.file_path,
                                          load_data=True,
                                          train=True,
                                          seed=args.seed,
                                          split=args.split,
                                          transform=args.transform),
                                 batch_size=1,
                                 shuffle=False,
                                 **loader_kwargs)




test_loader = data_utils.DataLoader(HDF5Dataset(file_path=args.file_path,
                                              load_data=True,
                                              train=False,
                                              seed=args.seed,
                                              split=args.split,
                                              transform=args.transform),
                                     batch_size=1,
                                     shuffle=False,
                                    **loader_kwargs)

print('Init Model')
model = Attention(ps=args.patch_size)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label, n) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.item()
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()
        #meep
        #empty_cache()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss, train_error))
    test()


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label, n) in enumerate(test_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.item()
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error
        
        n=n.data.cpu().numpy()
        label=label[0].data.cpu().numpy()
        predicted_label=predicted_label.data.cpu().numpy()
        print(str(n[0])+"\t"+str(label)+"\t"+str(int(predicted_label[0][0])))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print(' Test Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss, test_error))


if __name__ == "__main__":
    print('Start Training')
    print('# of training slides:\t' + str(len(train_loader)))
    print('# of testing slides:\t' + str(len(test_loader)))
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        #print(memory_cached())
    print('Start Testing')
    test()
    torch.save(model,str(args.seed)+'_base.model')
