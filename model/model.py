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
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
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
parser.add_argument('--x1-maxlen', type=int, default=150, metavar='N',
                    help='max length of symptom input')
parser.add_argument('--x2-maxlen', type=int, default=15, metavar='N',
                    help='max length of diagnose input')
parser.add_argument('--dimention', type=int, default=100, metavar='N',
                    help='length of vector dimention')
parser.add_argument('--sym-embedding-path', type=str, default='../../res/train_data_v3/symNpy.npy')
parser.add_argument('--diag-embedding-path', type=str, default='../../res/train_data_v3/mentionNpy.npy')
parser.add_argument('--train-path', type=str, default='../../res/train_data_v3/train_data.txt')
parser.add_argument('--test-path', type=str, default='../../res/train_data_v3/test_data.txt')


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
        self.w_tensor = nn.Linear(100, 100)
        self.u_tensor = nn.Linear(100, 100)
        self.v_tensor = nn.Linear(100, 1)

        # multi-task shared matrix
        self.w2_tensor = nn.Linear(100, 100)
        self.u2_tensor = nn.Linear(100, 100)
        self.v2_tensor = nn.Linear(100, 1)

        # mlp matrix
        #self.fc1 = []
        #self.fc2 = []
        #for i in range(15):
        self.fc1 = nn.Linear(200, 600)
        self.fc2 = nn.Linear(600, 588)

    def forward(self, x1, x2, x1_len, x2_len, Y):

        # x1: p*m  x2: q*m , padding with 0
        # x1_len: original length of x1    x2_len: original length of x2
        x1 = self.embedding_s(x1)

        # calculate W*S and extend to p*q*m
        w_tensor = self.w_tensor(x1)
        w_tensor = w_tensor.expand(15,args.batch_size,150,100)
        w_tensor = w_tensor.transpose(0,1)
        #print(w_tensor)

        # calculate U*D and extend to p*q*m
        x2 = self.embedding_d(x2)
        u_tensor = self.u_tensor(x2)
        u_tensor = u_tensor.expand(150,args.batch_size, 15, 100)
        u_tensor = u_tensor.transpose(0,1)
        u_tensor = u_tensor.transpose(1,2)

        # calculate local attention
        v_tensor = self.get_f_tensor(w_tensor, u_tensor)
        v_tensor = self.v_tensor(v_tensor)
        v_tensor = v_tensor.view(args.batch_size, 15, 150)
        input = self.get_local_attention(v_tensor, x1, x1_len, x2_len)

        # calculate W_2*C extend to q*q*m
        w2_tensor = self.w2_tensor(input)
        w2_tensor = w2_tensor.expand(15, args.batch_size, 15, 100)
        w2_tensor = w2_tensor.transpose(0,1)

        # calculate U_2*D extend to q*q*m
        u2_tensor = self.u2_tensor(x2)
        u2_tensor = u2_tensor.expand(15, args.batch_size, 15, 100)
        u2_tensor = u2_tensor.transpose(0,1)

        # calculate shared info
        v2_tensor = self.get_f_tensor(w2_tensor, u2_tensor)
        v2_tensor = self.v2_tensor(v2_tensor)
        v2_tensor = v2_tensor.view(args.batch_size, 15, 15)
        input_shared = self.get_shared_info(v2_tensor, input, x2, x2_len)

        # calculate mlp
        input_mlp = self.get_mlp_input(input, input_shared, x2, x2_len)
        output_tensor = []
        #print(input_mlp)

        #for i in range(15):
        input_fc1 = F.relu(self.fc1(input_mlp))
        output = F.relu(self.fc2(input_fc1))
        output = F.softmax(output)

        return output

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

    def get_idx_tensor(self, tensor_size, x1_len, x2_len, len1, len2):
        """
        Args:
            tensor_size: batch_size * p * q
            x1_len: original length of symptoms
            x2_len: original length of mentions
            return: idx_tensor, which x1_len * x2_len is 1, else is 0
        """
        idx_tensor = []
        x1_len_data = x1_len.data
        x2_len_data = x2_len.data
        for batch in range(args.batch_size):
            idx_numpy = [[0 for i in range(len1)] for j in range(len2)]
            for i in range(x2_len_data[batch]):
                for j in range(x1_len_data[batch]):
                    idx_numpy[i][j] = 1
            idx_tensor.append(idx_numpy)
        idx_tensor = torch.from_numpy(np.array(idx_tensor).astype(np.float)).float()
        return Variable(idx_tensor)

    def my_softmax(self, v_tensor, x1_len, x2_len):
        """
        Args:
            v_tensor: p * q * m
            return: p*q
        """
        v_tensor = torch.exp(v_tensor)
        idx_tensor = self.get_idx_tensor(v_tensor.size(), x1_len, x2_len, 150, 15)
        v_tensor = torch.mul(v_tensor, idx_tensor)
        v_tensor_sum = torch.sum(v_tensor, 2)
        v_tensor_sum = v_tensor_sum.expand(150, args.batch_size, 15)
        v_tensor_sum = v_tensor_sum.transpose(0, 1)
        v_tensor_sum = v_tensor_sum.transpose(1,2)
        alpha = torch.div(v_tensor, v_tensor_sum)
        alpha[(alpha != alpha).detach()] = 0
        return alpha

    def cal_attention(self, alpha, x, x1_len, x2_len):
        """
        Args:
            alpha: q * p
            x:  p * m
            return: q * m
        """
        # q*p * p*m = q*m
        output = torch.bmm(alpha, x)
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

    def get_diag(self, v_tensor):
        v_tensor_diag = []
        for i in range(args.batch_size):
            tensor_diag = torch.diag(v_tensor[i])
            v_tensor_diag.append(tensor_diag.data.numpy())
        return Variable(torch.from_numpy(np.array(v_tensor_diag)))

    def get_idx_tensor2(self, x2_len):
        idx_tensor = []
        x2_len_data = x2_len.data
        for batch in range(args.batch_size):
            idx = [0 for i in range(15)]
            for j in range(x2_len_data[batch]):
                idx[j] = 1
            idx_tensor.append(idx)
        idx_tensor = torch.from_numpy(np.array(idx_tensor).astype(np.float)).float()
        return Variable(idx_tensor)

    def my_softmax2(self, v_tensor, x2_len):
        """ compute weight of softmax """
        v_tensor_diag = self.get_diag(v_tensor)
        v_tensor_sum = torch.sum(v_tensor, 1)
        v_tensor = v_tensor_sum - v_tensor_diag
        v_tensor = torch.exp(v_tensor)
        idx_tensor = self.get_idx_tensor2(x2_len)
        v_tensor = torch.mul(v_tensor, idx_tensor)
        v_tensor_sum = torch.sum(v_tensor, 1)
        v_tensor_sum = v_tensor_sum.expand(15, args.batch_size)
        v_tensor_sum = v_tensor_sum.transpose(0,1)
        alpha = torch.div(v_tensor, v_tensor_sum)
        alpha[(alpha != alpha).detach()] = 0
        return alpha

    def cal_attention2(self, alpha, input, x2_len):
        """
        Args:
            alpha: batch_size * q
            input: batch_size * q * m
            x2_len: batch_size * 1
            return: batch_size * m
        """
        alpha = alpha.expand(1, args.batch_size, 15)
        alpha = alpha.transpose(0,1)
        output = torch.bmm(alpha, input)
        output = output.view(args.batch_size, 100)
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
        input_shared = input_shared.expand(15, args.batch_size, 100)
        input_shared = input_shared.transpose(0, 1)
        x = torch.cat((input, input_shared), 2)
        return x


# load symptom and diagnose embedding
sym_np = pickle.load(open(args.sym_embedding_path, 'r'))
sym_embedding = torch.from_numpy(sym_np.astype(np.double)).float()
diag_np = pickle.load(open(args.diag_embedding_path, 'r'))
diag_embedding = torch.from_numpy(diag_np.astype(np.double)).float()

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
                epoch, batch_idx * len(Y), len(train_loader.dataset),
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
