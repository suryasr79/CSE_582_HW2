# -*- coding:utf8 -*-
"""
This py page is for the Modeling and training part of this NLM.
Try to edit the place labeled "# TODO"!!!

Modifications:
1. add a command line argument specifying the path to the movies or jewelry dataset
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import time
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONTEXT_SIZE = 2
EMBEDDING_DIM = 50
BATCH_SIZE = 256

trigrams = []
vocab = {}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, vocab, trigrams):
        self.trigrams = trigrams
        self.vocab = vocab

    def __len__(self):
        return len(self.trigrams)

    def __getitem__(self, index):
        target_trigram = self.trigrams[index]
        contexts, label = target_trigram
        context_0, context_1, label = self.vocab[contexts[0]], self.vocab[contexts[1]], self.vocab[label]
        return torch.LongTensor([context_0, context_1]), label


def word2index(word, vocab):
    """
    Convert an word token to an dictionary index
    """
    if word in vocab:
        value = vocab[word]
    else:
        value = -1
    return value


def index2word(index, vocab):
    """
    Convert an word index to a word token
    """
    for w, v in vocab.items():
        if v == index:
            return w
    return 0


def preprocess(file, is_filter=True):
    """
    Prepare the data and the vocab for the models.
    For expediency, the vocabulary will be all the words
    in the dataset (not split into training/test), so
    the assignment can avoid the OOV problem.
    """
    with open(file, 'r') as fr:
        for idx, line in enumerate(fr):
            line = line.strip()
            text = line.split(',', 1)[1]
            words = word_tokenize(text)
            if is_filter:
                words = [w for w in words if not w in stop_words]
                words = [word.lower() for word in words if word.isalpha()]
                for word in words:
                    if word not in vocab:
                        vocab[word] = len(vocab)
            if len(words) > 0:
                for i in range(len(words) - 2):
                    trigrams.append(([words[i], words[i + 1]], words[i + 2]))
    print('{0} contain {1} lines'.format(file, idx + 1))
    print('The size of dictionary is：{}'.format(len(vocab)))
    print('The size of trigrams is：{}'.format(len(trigrams)))
    return vocab, trigrams


class NgramLM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NgramLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 256)
        self.linear2 = nn.Linear(256, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))  # Assuming batch size 1
        # TODO create the final layer, and calculate the log probabilities of the classes
        out =
        log_probs =
        return log_probs


def output(model, file1, file2):
    """
    Write the embedding file and randomly initialized vectors to the data folder
    """

    with open(file1, 'w') as fw1:
        for word, id in vocab.items():
    # TODO Use the model to convert words to embeddings and write to file1
    with open(file2, 'w') as fw2:
        for word, id in vocab.items():
    # TODO Initialize some random embeddings and write to file2


def training(trainfile):
    """
    Train the NLM
    """
    vocab, trigrams = preprocess(trainfile)
    # Determine how large a file can be used for training, for either dataset, and prepare the datasets
    if os.path.split(trainfile)[-1] == 'movies_train.csv':
        outfile = os.path.join('.', 'data', 'movie_embeddings.txt')
    else:
        outfile = os.path.join('.', 'data', 'jewelry_embeddings.txt')
    dataset = Dataset(vocab, trigrams)
    # set num_workers to 0 to avoid error messages during training per this stackflow post:
    # https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
    generator = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    # generator = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    losses = []
    model = NgramLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    model.to(device)
    model.train()
    # TODO Choose your optimizer and loss function
    optimizer =
    loss_function =
    for epoch in range(100):  # you can increase the epochs but only if you stay within the instructed time limits
        total_loss = 0
        print(epoch)
        for context, target in generator:
            # Step 1. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old instance
            # TODO
            # Step 2. Run the forward pass, getting log probabilities over next words
            # TODO
            # Step 3. Compute your loss function.
            # TODO
            # Step 4. Do the backward pass and update the gradient
            # TODO
            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
        print(total_loss)
    print(losses)  # The loss decreased every iteration over the training data!
    model = model.to('cpu')
    output(model, outfile, './data/embedding_random_500.txt')


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Text Classification')
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_file', default='./data/movies/movies_train.txt',
                               help='Path of the input reviews text')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainfile = args.train_file
    start_time = time.time()
    training(trainfile)
    print('Total time: {:.4f}'.format(time.time() - start_time))
