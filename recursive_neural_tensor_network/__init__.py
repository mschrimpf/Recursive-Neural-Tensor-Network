# coding: utf-8

# # 9. Recursive Neural Networks and Constituency Parsing

# I recommend you take a look at these material first.

# * http://web.stanford.edu/class/cs224n/lectures/cs224n-2017-lecture14-TreeRNNs.pdf
# * https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf
import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import Image, display
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from torch.autograd import Variable

from recursive_neural_tensor_network.data import load_data, get_batch
from recursive_neural_tensor_network.model import Tree, RNTN
from recursive_neural_tensor_network.utils import flatten

USE_CUDA = torch.cuda.is_available()


class Defaults(object):
    seed = 1024
    hidden_size = 30
    root_only = False
    batch_size = 20
    epoch = 20
    learning_rate = 0.01
    lambda_ = 1e-5
    rescheduled = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=Defaults.seed)
    parser.add_argument('--hidden_size', type=int, default=Defaults.hidden_size)
    parser.add_argument('--root_only', action='store_true', default=Defaults.root_only)
    parser.add_argument('--no-root_only', action='store_false', dest='root_only')
    args = parser.parse_args()

    random.seed(args.seed)

    dataset_directory = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..',
        'neural-nlp', 'ressources', 'data', 'stanford-sentiment-treebank', 'trees'))

    # gpus = [0]
    # torch.cuda.set_device(gpus[0])

    # ## Data
    sample = random.choice(open(os.path.join(dataset_directory, 'train.txt'), 'r', encoding='utf-8').readlines())
    print(sample)
    train_data, word2index = load_data(dataset_directory=dataset_directory, type='train')

    # ## Modeling
    model = RNTN(word2index, args.hidden_size, 5)
    model.init_weight()
    if USE_CUDA:
        model = model.cuda()

    # ## Training
    train_epochs(model, train_data)
    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'weights'))

    # ## Test
    test_data, _ = load_data(dataset_directory=dataset_directory, type='test')
    accuracy = compute_accuracy(model, test_data, ROOT_ONLY=args.root_only)
    print(accuracy)


def compute_accuracy(model, data, ROOT_ONLY):
    accuracy = 0
    num_node = 0
    for test in data:
        model.zero_grad()
        preds = model(test, ROOT_ONLY)
        labels = test.labels[-1:] if ROOT_ONLY else test.labels
        for pred, label in zip(preds.max(1)[1].data.tolist(), labels):
            num_node += 1
            if pred == label:
                accuracy += 1
    accuracy = accuracy / num_node * 100
    return accuracy


def train_epochs(model, train_data,
                 batch_size=Defaults.batch_size, num_epochs=Defaults.epoch, learning_rate=Defaults.learning_rate,
                 lambda_=Defaults.lambda_, rescheduled=Defaults.rescheduled, root_only=Defaults.root_only):
    # It takes for a while... It builds its computational graph dynamically. So Its computation is difficult to train with batch.

    LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        losses = []

        # learning rate annealing
        if rescheduled == False and epoch == num_epochs // 2:
            learning_rate *= 0.1
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_)  # L2 norm
            rescheduled = True

        for i, batch in enumerate(get_batch(batch_size, train_data)):

            if root_only:
                labels = [tree.labels[-1] for tree in batch]
                labels = Variable(LongTensor(labels))
            else:
                labels = [tree.labels for tree in batch]
                labels = Variable(LongTensor(flatten(labels)))

            model.zero_grad()
            preds = model(batch, root_only)

            loss = loss_function(preds, labels)
            losses.append(loss.data.tolist()[0])

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('[%d/%d] mean_loss : %.2f' % (epoch, num_epochs, np.mean(losses)))
                losses = []
    # The convergence of the model is unstable according to the initial values. I tried to 5~6 times for this.


def draw_nltk_tree(tree):
    # Borrowed from
    # https://stackoverflow.com/questions/31779707/how-do-you-make-nltk-draw-trees-that-are-inline-in-ipython-jupyter

    cf = CanvasFrame()
    tc = TreeWidget(cf.canvas(), tree)
    tc['node_font'] = 'arial 15 bold'
    tc['leaf_font'] = 'arial 15'
    tc['node_color'] = '#005990'
    tc['leaf_color'] = '#3F8F57'
    tc['line_color'] = '#175252'
    cf.add_widget(tc, 50, 50)
    cf.print_to_file('tmp_tree_output.ps')
    cf.destroy()
    os.system('convert tmp_tree_output.ps tmp_tree_output.png')
    display(Image(filename='tmp_tree_output.png'))
    os.system('rm tmp_tree_output.ps tmp_tree_output.png')


if __name__ == '__main__':
    main()
