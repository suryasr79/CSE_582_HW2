# -*- coding:utf8 -*-

import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from classifier import Model
from collections import Counter

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(file):
    """
    Prepare the labeled data and the vocabulary for the model. For convenience,
    as noted in the instructions, the vocabulary will be all the words
    in training + test: this avoids out-of-vocab (OOV) words.
    """
    all_words = []
    labels = []
    sentences = []
    with open(file, 'r') as fr:
        for idx, line in enumerate(fr):
            trainrow = line.strip()
            label = trainrow.split(',', 1)[0]
            labels.append(int(label))
            text = trainrow.split(',', 1)[1]
            words = [w.lower() for w in word_tokenize(text) if w not in stop_words]
            words = [w for w in words if w.isalpha()]
            if len(words) > 0:
                all_words += words
                sentences.append(words)
    print('{0} contains {1} lines, {2} words.'.format(file, idx + 1, len(all_words)))
    return labels, sentences, all_words

def modeling(args, trainlabels, trainsentences, testlabels, testsentences, vocab):
    """
    Initialize the model, train and test for some number of epochs, depending on the datarep
    (BOW or GLOVE); per the instructions, you need to specify the number of training epochs for
    model.training()
    """
    model = Model(args, vocab, trainlabels, trainsentences, testlabels, testsentences)
    model.prepare_datasets()
    model.training()
    return 0

def construct_vocab_dict(args, bagofwords):
    vocab = {}
    cnt = Counter(bagofwords)
    for word, freq in cnt.items():
        # Note that words whose frequency is less than 3 are excluded if --alpha True.
        # Again, for expedience we (incorrectly) use the entire dataset to define the vocabulary),
        # to avoid the case of OOV (out-of-vocabulary) words in the test data
        if args.alpha == 'True':
            if (freq > 3):
                vocab[word] = [len(vocab), freq]
        else:
            vocab[word] = [len(vocab), freq]
    return vocab

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Text Classification')
    parser.add_argument('--train', action='store_true',
                        help='if use the whole dataset')
    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--datarep', choices=['BOW', 'GLOVE'], default='BOW',
                                help='choose the input word vector representation to use')
    model_settings.add_argument('--lr', type=float, default=0.0001,
                                help='learning rate')
    model_settings.add_argument('--hidden_size', type=int, default=5,
                                help='the hidden size of the classifier')
    model_settings.add_argument('--embed_size', type=int, default=50,
                                help='size of the glove embeddings')
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--embed_file', default=['./glove/glove.6B.50d.txt'],
                               help='Path of pre-trained input data')
    path_settings.add_argument('--train_file', default='./data/movies/movies_train.txt',
                               help='Path of the input reviews text')
    path_settings.add_argument('--test_file', default='./data/movies/movies_test.txt',
                               help='Path of the input reviews label')
    path_settings.add_argument('--alpha', default='True',
                            help='Whether to drop non-words')
    return parser.parse_args()


def run():
    """
    Prepares the data, constructs the vocabulary dictionary, then trains and tests the model.
    """
    args = parse_args()
    trainlabels, trainsentences, trainwords = preprocess(args.train_file)
    testlabels, testsentences, testwords = preprocess(args.test_file)

    final_words = trainwords + testwords
    print("Size of entire corpus in words: ",len(final_words)," from train ",len(trainwords)," and test ",len(testwords))
    # NOTE: to avoid out-of-vocabulary (OOV) words at test time, for convenience we create
    # a dictionary of all words in train and test, although it is more correct to handle OOV words
    # Build vocab : {w:[id, frequency]}
    final_words_set = set(final_words)
    print("Vocab size from all words before counter:", len(final_words_set),"out of ",len(final_words))
    vocab = construct_vocab_dict(args, final_words)
    print('The vocabulary size is：{}'.format(len(vocab)))
    modeling(args, trainlabels, trainsentences, testlabels, testsentences, vocab)

"""
Sample calls:

python run.py --datarep BOW --lr 0.001 --hidden_size 100 --train_file ./data/movies/movies_train.csv --test_file ./data/movies/movies_test.csv

python run.py --datarep GLOVE --lr 0.0001 --hidden_size 200 --embed_size 300 --embed_file ./glove/glove.6B.300d.txt --train_file ./data/movies/movies_train.csv --test_file ./data/movies/movies_test.csv 
"""
if __name__ == '__main__':
    run()
