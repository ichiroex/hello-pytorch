import argparse
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

""" XOR演算子の学習
"""
data_list = [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
label_list = [[0], [1], [1], [0]]

NEPOCH = 100000000

class Model(nn.Module):

    def __init__(self, hidden=10):
        super(Model, self).__init__()
        self.l1 = nn.Linear(2, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, input):
        h = F.relu(self.l1(input))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y


def train():

    model = Model()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(NEPOCH):

        data = Variable(torch.FloatTensor(data_list))
        target = Variable(torch.FloatTensor(label_list))

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print "epoch:%d" % epoch, "loss:%5f" % loss.data[0]
            # TEST
            test_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
            random.shuffle(test_data)
            tdata = Variable(torch.FloatTensor(test_data))
            pred = model(tdata)
            print "input: ", test_data
            print "predi: ", pred.data
            print " - - - - - - -"


def main():
    train()

if __name__ == "__main__":
    main()
