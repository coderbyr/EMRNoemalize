# coding: utf-8

"""
implemented pytorch version
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cPickle as pickle
import torch.optim as optim
from torch.autograd import Variable
from data_loader import get_loader


parser = argparse.ArgumentParser(description='PyTorch Model Parameters')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='w',
                    help='input batch size for testing (default: 50)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--x1-maxlen', type=int, default=50, metavar='N',
                    help='max length of symptom input')
parser.add_argument('--x2-maxlen', type=int, default=10, metavar='N',
                    help='max length of diagnose input')
parser.add_argument('--dimention', type=int, default=100, metavar='N',
                    help='length of vector dimention')
parser.add_argument('--sym-embedding-path', type=str, default='../../res/train_data_v2/sym_vec.npy')
parser.add_argument('--diag-embedding-path', type=str, default='../../res/train_data_v2/diag_vec.npy')
parser.add_argument('--train-path', type=str, default='../../res/train_data_v2/train_data.txt')
parser.add_argument('--test-path', type=str, default='../../res/train_data_v2/test_data.txt')


args = parser.parse_args()

kwargs = {}

class DynamicModel(nn.Module):

    def __init__(self, sym_embedding, diag_embedding):
        super(DynamicModel, self).__init__()
        self.embedding_s = nn.Embedding(sym_embedding.size(0), sym_embedding.size(1))
        self.embedding_s.weight = nn.Parameter(sym_embedding)
        self.embedding_d = nn.Embedding(diag_embedding.size(0), sym_embedding.size(1))
        self.embedding_d.weight = nn.Parameter(diag_embedding)

        # model parameters
        # local attention matrix
        self.w_tensor = nn.Linear(50*100, 50*100)
        self.u_tensor = nn.Linear(10*100, 10*100)
        self.v_tensor = nn.Linear(50*10, 50*10)

        # multi-task shared matrix
        self.w2_tensor = nn.Linear(10*100, 10*100)
        self.u2_tensor = nn.Linear(10*100, 10*100)
        self.v2_tensor = nn.Linear(10*10, 10*10)

        # mlp matrix
        self.fc1 = []
        self.fc2 = []
        for i in range(10):
            self.fc1.append(nn.Linear(3*100, 1000))
            self.fc2.append(nn.Linear(1000, 800))

    def forward(self, x1, x2, x1_len, x2_len, Y):

        # x1: p*m  x2: q*m , padding with 0
        # x1_len: original length of x1    x2_len: original length of x2
        x1 = self.embedding_s(x1)
        print(x1[0])

        # calculate W*S and extend to p*q*m
        w_tensor = self.w_tensor(x1)
        w_tensor = w_tensor.expand(50,100,10)

        # calculate U*D and extend to p*q*m
        u_tensor = self.u_tensor(x2)
        u_tensor = u_tensor.expand(10,100,50)
        u_tensor = u_tensor.transpose(0,2)

        # calculate local attention
        v_tensor = self.get_f_tensor(w_tensor, u_tensor)
        v_tensor = self.v_tensor(v_tensor)
        input = self.get_local_attention(v_tensor, x1, x1_len, x2_len)

        # calculate W_2*C extend to q*q*m
        w2_tensor = self.w2_tensor(input)
        w2_tensor = w2_tensor.expand(10, 100, 10)
        w2_tensor = w2_tensor.transpose(1,2)

        # calculate U_2*D extend to q*q*m
        u2_tensor = self.u2_tensor(x2)
        u2_tensor = u2_tensor.expand(10, 100, 10)
        u2_tensor = u2_tensor.transpose(1,2)

        # calculate shared info
        v2_tensor = self.get_f_tensor(w2_tensor, u2_tensor)
        v2_tensor = self.v2_tensor(v2_tensor)
        input_shared = self.get_shared_info(v2_tensor, input, x2, x2_len)

        # calculate mlp
        input_mlp = self.get_mlp_input(input, input_shared, x2, x2_len)
        output_tensor = []
        for i in range(10):
            input_fc1 = F.relu(self.fc1[i](input_mlp[:(i+1)*3, :]))
            output = F.relu(self.fc2[i](input_fc1))
            output = F.softmax(output)
            output_tensor.append(output)

        return output_tensor

    def get_f_tensor(self, w_tensor, u_tensor):
        """
        Args:
            w_tensor: p * q * m
            u_tensor: p * q * m
            return: p * q * m
        """
        f_tensor = w_tensor + u_tensor
        f_tensor = F.tanh(f_tensor)
        return f_tensor

    def my_softmax(self, v_tensor, x1_len, x2_len):
        """
        Args:
            v_tensor: p * q * m
            return: p*q
        """
        v_tensor = v_tensor[:x1_len, :x2_len]
        v_tensor = F.softmax(v_tensor)
        v_tensor = v_tensor.expand(50, 10)
        return v_tensor

    def cal_attention(self, alpha, x, x1_len, x2_len):
        """
        Args:
            alpha: p * q
            x:  p * m
            return: x2_len * m
        """
        x = x[:x1_len, :]
        alpha = alpha[:x1_len, :x2_len]
        alpha = alpha.transpose(0,1)
        output = torch.matmul(alpha, x)
        return output

    def get_local_attention(self, v_tensor, x1, x1_len, x2_len):
        """
        Args:
            v_tensor: p*q
            x1: p*m
            return: q*m
        """
        # alpha size: p*q
        alpha = self.my_softmax(v_tensor, x1_len, x2_len)

        # local attention size: q * m
        output = self.cal_attention(alpha, x1, x1_len, x2_len)
        return output

    def my_softmax2(self, v_tensor, x2_len):
        v_tensor = v_tensor[:x2_len, :x2_len]
        v_tensor = torch.Tensor([v_tensor[z].sum() - v_tensor[z][z] for z in range(len(v_tensor))])
        v_tensor = F.softmax(v_tensor)
        v_tensor = v_tensor.expand(10)
        return v_tensor

    def cal_attention2(self, alpha, input, x2_len):
        x = input[:x2_len, :]
        alpha = alpha[:x2_len]
        output = torch.matmul(alpha, x)
        return output

    def get_shared_info(self, v_tensor, input, x2, x2_len):
        """
        Args:
            v_tensor: q*q
            input: q*m
            return: m
        """
        alpha = self.my_softmax2(v_tensor, x2_len)
        output = self.cal_attention2(alpha, input, x2_len)
        return output

    def get_mlp_input(self, input, input_shared, x2, x2_len):
        x = input + input_shared
        x += x2
        return x


# load symptom and diagnose embedding
sym_np = pickle.load(open(args.sym_embedding_path, 'r'))
sym_embedding = torch.from_numpy(sym_np.astype(np.double))
diag_np = pickle.load(open(args.diag_embedding_path, 'r'))
diag_embedding = torch.from_numpy(diag_np.astype(np.double))

model = DynamicModel(sym_embedding, diag_embedding)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loader = get_loader(args.train_path)
test_loader = get_loader(args.test_path, 1)

def my_criterion(output, target):
    loss = 0
    for i in range(len(output)):
        loss += criterion(output[i], target[i])
    return loss

def train(epoch):
    model.train()
    for batch_idx, (symptoms, mentions, entities, sym_lens, men_lens) in enumerate(train_loader):
        x1, x2, Y, x1_len, x2_len = Variable(symptoms), Variable(mentions), Variable(entities),\
                                    Variable(sym_lens), Variable(men_lens)
        optimizer.zero_grad()
        output = model(x1, x2, x1_len, x2_len, Y)
        loss = my_criterion(output, Y)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch : {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0])
            )
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (symptoms, mentions, entities, sym_lens, men_lens) in enumerate(test_loader):
        x1, x2, Y, x1_len, x2_len = Variable(symptoms, volatile=True), Variable(mentions, volatile=True), \
                                    Variable(entities), Variable(sym_lens, volatile=True), \
                                    Variable(men_lens, volatile=True)
        output = model(x1, x2, x1_len, x2_len, Y)
        test_loss += my_criterion(output, Y).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(test.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
