# coding: utf-8

import torch
from torch import nn
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)


for epoch in range(10):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print "\tgrad:", x, y, w.grad.data[0]
        w.data = w.data - 0.01 * w.grad.data
        w.grad.data.zero_()
        print "progress:", epoch, l.data[0]