# -*- coding:utf8 -*-

import torch
import torch.nn as nn
import torch.optim
import numpy as np

def word2index(word, vocab):
    """
    Convert an word token to an dictionary index
    """
    if word in vocab:
        value = vocab[word][0]
    else:
        value = -1
    return value

def index2word(index, vocab):
    """
    Convert an word index to a word token
    """
    for w, v in vocab.items():
        if v[0] == index:
            return w
    return 0

class Model(object):
    def __init__(self, args, vocab, trainlabels, trainsentences, testlabels, testsentences):
        """ The Text Classification model constructor """
        self.embeddings_dict = {}
        self.datarep = args.datarep
        if self.datarep == "GLOVE":
            print("Now we are using the GloVe embeddings")
            self.load_glove(args.embed_file)
        else:
            print("Now we are using the BOW representation")
        self.vocab = vocab
        self.trainlabels = trainlabels
        self.trainsentences = trainsentences
        self.testlabels = testlabels
        self.testsentences = testsentences
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.hidden_size =args.hidden_size
        self.traindataset = []
        self.testdataset = []

        if self.datarep == "GLOVE":
            self.model = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
            nn.LogSoftmax(dim=1),)
        else:
            self.model = nn.Sequential(
                nn.Linear(len(vocab), self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 2),
                nn.LogSoftmax(dim=1), )

    def prepare_datasets(self):
        """
        Load both training and test
        Convert the text spans to BOW or GLOVE vectors
        """

        datasetcount = 0

        for setOfsentences in [self.trainsentences, self.testsentences]:

            sentcount = 0
            datasetcount += 1

            for sentence in setOfsentences:
                sentcount += 1
                # vsentence holds lexical (GLOVE) or word index (BOW) input to sentence2vec
                vsentence = []
                for l in sentence:
                    if l in self.vocab:
                        if self.datarep == "GLOVE":
                            vsentence.append(l)
                        # for now this can remain, but when tfidf is used, add more conditions
                        else:
                            vsentence.append(word2index(l, self.vocab))
                svector = self.sentence2vec(vsentence, self.vocab)
                if (len(vsentence) > 0) & (datasetcount == 1): # train
                    self.traindataset.append(svector)
                elif (len(vsentence) > 0) & (datasetcount == 2): # test
                    self.testdataset.append(svector)

        print("\nDataset size for train: ",len(self.traindataset)," out of ",len(self.trainsentences))
        print("\nDataset size for test: ",len(self.testdataset)," out of ",len(self.testsentences))
        indices = np.random.permutation(len(self.traindataset))

        self.traindataset = [self.traindataset[i] for i in indices]
        self.trainlabels = [self.trainlabels[i] for i in indices]
        self.trainsentences = [self.trainsentences[i] for i in indices]

    def rightness(self, predictions, labels):
        """ 
        Prediction of the error rate
        """
        pred = torch.max(predictions.data, 1)[1]  #
        rights = pred.eq(labels.data.view_as(pred)).sum()  #
        return rights, len(labels)  #

    def sentence2vec(self, sentence, dictionary):
        """ 
        Convert sentence text to vector representation 
        """
        if self.datarep == "GLOVE":
            vector = np.zeros(self.embed_size)
            for word in sentence:
                if word in self.embeddings_dict:
                    vector += self.embeddings_dict[word]
            return 1.0 * vector / len(sentence)
        else:
            vector = np.zeros(len(dictionary))
            for l in sentence:
                vector[l] += 1
            return 1.0 * vector / len(sentence)

    def load_glove(self, path):
        """
        Load Glove embeddings dictionary
        """
        with open(path, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        return 0

    def training(self):
        """
        The whole training and testing process.
        """
        losses = []
        loss_function = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Change the number of training epochs here
        if self.datarep == "GLOVE":
            tr_epochs = 20
        else:
            tr_epochs = 10

        for epoch in range(tr_epochs):
            print(epoch)
            for i, data in enumerate(zip(self.traindataset, self.trainlabels)):
                x, y = data
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                optimizer.zero_grad()
                # predict
                predict = self.model(x)
                # calculate loss
                loss = loss_function(predict, y)
                losses.append(loss.data.numpy())
                loss.backward()
                optimizer.step()
                # test every 1000 epoch
                if i % 1000 == 0:
                    val_losses = []
                    rights = []
                    for j, test in enumerate(zip(self.testdataset, self.testlabels)):
                        x, y = test
                        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                        y = torch.tensor(np.array([y]), dtype=torch.long)
                        predict = self.model(x)
                        right = self.rightness(predict, y)
                        rights.append(right)
                        loss = loss_function(predict, y)
                        val_losses.append(loss.data.numpy())

                    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                    print('At the {} epoch，Training loss：{:.2f}, Testing loss：{:.2f}, Testing Acc: {:.2f}'.format(epoch, np.mean(losses),
                                                                                np.mean(val_losses), right_ratio))
        print("Training End")
