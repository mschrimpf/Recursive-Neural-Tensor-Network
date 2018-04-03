import os
import random

from recursive_neural_tensor_network.utils import flatten
from recursive_neural_tensor_network.model import Tree


def loadTrees(dataset_directory, type='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    file = os.path.join(dataset_directory, '%s.txt' % type)
    print("Loading %s trees.." % type)
    with open(file, 'r', encoding='utf-8') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    return trees


def load_data(dataset_directory, type='train'):
    # ## Data load and Preprocessing
    # ### Stanford Sentiment Treebank(https://nlp.stanford.edu/sentiment/index.html)

    # draw_nltk_tree(nltkTree.fromstring(sample))

    train_data = loadTrees(dataset_directory=dataset_directory, type=type)
    # ### Build Vocab
    vocab = list(set(flatten([t.get_words() for t in train_data])))
    word2index = {'<UNK>': 0}
    for vo in vocab:
        if word2index.get(vo) is None:
            word2index[vo] = len(word2index)
    index2word = {v: k for k, v in word2index.items()}
    return train_data, word2index


def get_batch(batch_size, data):
    random.shuffle(data)
    sindex = 0
    eindex = batch_size
    while eindex < len(data):
        batch = data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(data):
        batch = data[sindex:]
        yield batch
