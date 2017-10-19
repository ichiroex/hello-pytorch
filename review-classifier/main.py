# -*- coding: UTF-8 -*-
import argparse
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import Counter

""" XOR演算子の学習
"""
data_list = [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
label_list = [[0], [1], [1], [0]]

NEPOCH = 100000000

parser = argparse.ArgumentParser()
parser.add_argument("fname", type=str, help="Filename of dataset")
args = parser.parse_args()

dataset = []
with open(args.fname, "r") as f:
    for line in f:
        dataset.append(line.strip().split(" ", 1))

N = len(dataset)
TRAIN_DATA_RATE = 0.8
VALID_DATA_RATE = 0.1
TEST_DATA_RATE  = 0.1

# データ分割
train_dataset = np.array(dataset[:int(N * TRAIN_DATA_RATE)])
valid_dataset = np.array(dataset[len(train_dataset):(len(train_dataset)+int(N * VALID_DATA_RATE))])
test_dataset  = np.array(dataset[(len(train_dataset)+len(valid_dataset)):])

# ラベルと文に分割
train_labels, train_sens = np.hsplit(train_dataset, [1])
valid_labels, valid_sens = np.hsplit(valid_dataset, [1])
test_labels, test_sens = np.hsplit(test_dataset, [1])

print "DATASET SIZE:", N
print "TRAIN DATA SIZE:", len(train_dataset)
print "VALID DATA SIZE:", len(valid_dataset)
print "TEST DATA SIZE:", len(test_dataset)

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


class Dict:
    def __init__(self, dataset, vocab):
        """ 辞書管理クラス
        Args:
            dataset: 文リスト
            vocab: 語彙数
        Return:
        """

        self.dataset = dataset

        counter = Counter()
        # WORD COUNT via Counter
        all_words = []
        for sen in self.dataset:
            all_words += sen[0].split()
        counter = Counter(all_words)

        print "Original vocab:", counter.__len__()
        print "Number of vocab:", vocab

        self.dic_word2id = {}
        self.dic_word2id["<pad>"] = 0
        self.dic_word2id["<s>"] = 1
        self.dic_word2id["</s>"] = 2
        self.dic_word2id["<unk>"] = 3

        self.dic_id2word = {}
        self.dic_id2word[0] = "<pad>"
        self.dic_id2word[1] = "<s>"
        self.dic_id2word[2] = "</s>"
        self.dic_id2word[3] = "<unk>"

        N_DUMMY = 4

        for idx, (word, num) in enumerate(counter.most_common()):
            # 語彙数制限
            if idx > vocab:
                break
            self.dic_word2id[word] = idx + N_DUMMY
            self.dic_id2word[idx + N_DUMMY] = word

    def word2id(self, word):
        """
        Args:
            word: word
        Return:
            id: id correspoding to input word
        """
        return self.dic_word2id.get(word, 3)

    def id2word(self, id):
        """
        Args:
            id: word id
        Return:
            word: a word correspoding to input id
        """
        return self.dic_id2word.get(id, "<unk>")


def train():

    # 辞書作成
    dic = Dict(train_sens, 5000)

    exit()
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
