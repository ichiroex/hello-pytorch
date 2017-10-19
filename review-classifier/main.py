# -*- coding: UTF-8 -*-
import argparse
import six
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

""" レビュー分類
"""
data_list = [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
label_list = [[0], [1], [1], [0]]

NEPOCH = 10

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

    def __init__(self, vocab):
        super(Model, self).__init__()
        self.emb = nn.Embedding(vocab, 300, padding_idx=0)
        self.rnn = nn.LSTM(300, 500)
        self.l1 = nn.Linear(500, 100)
        self.l2 = nn.Linear(100, 1)

    def forward(self, input):
        h = self.emb(input)
        h = F.relu(self.rnn(h))
        h = F.relu(self.l1(h))
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

        self.counter = Counter()
        # WORD COUNT via Counter
        all_words = []
        for sen in self.dataset:
            all_words += sen[0].split()
        self.counter = Counter(all_words)

        print "Original vocab:", self.counter.__len__()
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

        for idx, (word, num) in enumerate(self.counter.most_common()):
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


def sentence2id(sentence, dic):
    """ convert a sentence into id list using word dictionary
    """
    return [dic.word2id("<s>")] + [dic.word2id(word) for word in sentence.split()] + [dic.word2id("</s>")]

def fill_batch(sen_list):
    max_len = max([len(sen) for sen in sen_list])

    filled_sen_list = []
    for sen in sen_list:
        filled_sen_list.append(sen + [0 for _ in range(max_len-len(sen))])

    return filled_sen_list

def train():

    # 辞書作成
    dic = Dict(train_sens, 5000)

    # 言語データをWordID化
    train_sens_id = np.array([sentence2id(sentence[0], dic) for sentence in train_sens])

    N = len(train_sens_id) # 事例数
    batchsize = 50

    model = Model(5000 + 4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(NEPOCH):

        # training
        perm = np.random.permutation(N) #ランダムな整数列リストを取得
        sum_loss     = 0.0
        sum_accuracy = 0.0

        for i in six.moves.range(0, N, batchsize):

            x = Variable(torch.FloatTensor(fill_batch(train_sens_id[perm[i:i + batchsize]]))) # source
            t = Variable(torch.FloatTensor(np.array(train_labels[perm[i:i + batchsize]], dtype=np.float16).tolist()))  # target

            optimizer.zero_grad()
            y = model(x)
            loss = criterion(y, t)
            loss.backward()
            optimizer.step()
            print "epoch:%d" % epoch, "loss:%5f" % loss.data[0]

def main():
    train()

if __name__ == "__main__":
    main()
