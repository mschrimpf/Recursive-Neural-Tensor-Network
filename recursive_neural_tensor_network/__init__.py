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
    num_epochs = 20
    learning_rate = 0.01
    lambda_ = 1e-5
    rescheduled = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_directory', type=str, required=True)
    parser.add_argument('--seed', type=int, default=Defaults.seed)
    parser.add_argument('--hidden_size', type=int, default=Defaults.hidden_size)
    parser.add_argument('--num_epochs', type=int, default=Defaults.num_epochs)
    parser.add_argument('--batch_size', type=int, default=Defaults.batch_size)
    parser.add_argument('--learning_rate', type=float, default=Defaults.learning_rate)
    parser.add_argument('--lambda', type=float, dest='lambda_', default=Defaults.lambda_)
    parser.add_argument('--rescheduled', action='store_true', default=Defaults.rescheduled)
    parser.add_argument('--no-rescheduled', action='store_false', dest='rescheduled')
    parser.add_argument('--root_only', action='store_true', default=Defaults.root_only)
    parser.add_argument('--no-root_only', action='store_false', dest='root_only')
    args = parser.parse_args()
    print("Running with args {}".format(vars(args)))

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    sample = random.choice(open(os.path.join(args.dataset_directory, 'train.txt'), 'r', encoding='utf-8').readlines())
    print("Sample from the train set:", sample)
    train_data, word2index = load_data(dataset_directory=args.dataset_directory, type='train')
    test_data, _ = load_data(dataset_directory=args.dataset_directory, type='test')

    # Model
    model = RNTN(word2index, args.hidden_size, 5)
    model.init_weight()
    if USE_CUDA:
        model = model.cuda()
    weights_dir = os.path.dirname(__file__)

    # Run
    def save_hook(epoch, model):
        torch.save(model.state_dict(), os.path.join(weights_dir, 'weights-epoch{}'.format(epoch)))

    def test_accuracy_hook(epoch, model):
        accuracy = compute_accuracy(model, test_data, root_only=args.root_only)
        print("Test accuracy: ".format(accuracy))

    print("Starting training")
    train_epochs(model, train_data,
                 num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                 lambda_=args.lambda_, rescheduled=args.rescheduled, root_only=args.root_only,
                 post_epoch_hooks=[save_hook, test_accuracy_hook])


def train_epochs(model, train_data,
                 batch_size=Defaults.batch_size, num_epochs=Defaults.num_epochs, learning_rate=Defaults.learning_rate,
                 lambda_=Defaults.lambda_, rescheduled=Defaults.rescheduled, root_only=Defaults.root_only,
                 post_epoch_hooks=()):
    # It takes for a while... It builds its computational graph dynamically.
    # So Its computation is difficult to train with batch.
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, num_epochs + 1):
        # learning rate annealing
        if rescheduled is False and epoch == num_epochs // 2:
            learning_rate *= 0.1
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_)  # L2 norm
            rescheduled = True
        batch_losses = train_epoch(model, train_data, optimizer, batch_size, loss_function, root_only)
        print('[%d/%d] mean_loss : %.2f' % (epoch, num_epochs, np.mean(batch_losses)))
        for post_epoch_hook in post_epoch_hooks:
            post_epoch_hook(epoch, model)
    # The convergence of the model is unstable according to the initial values. I tried to 5~6 times for this.


def train_epoch(model, train_data, optimizer, batch_size, loss_function, root_only):
    losses = []
    for batch in get_batch(batch_size, train_data):
        loss = train_batch(model, batch, loss_function, optimizer, root_only)
        losses.append(loss.data.tolist()[0])
    return losses


def train_batch(model, batch, loss_function, optimizer, root_only):
    LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
    if root_only:
        labels = [tree.labels[-1] for tree in batch]
        labels = Variable(LongTensor(labels))
    else:
        labels = [tree.labels for tree in batch]
        labels = Variable(LongTensor(flatten(labels)))
    model.zero_grad()
    preds = model(batch, root_only)
    loss = loss_function(preds, labels)
    loss.backward()
    optimizer.step()
    return loss


def compute_accuracy(model, data, root_only):
    accuracy = 0
    num_node = 0
    for test in data:
        model.zero_grad()
        preds = model(test, root_only)
        labels = test.labels[-1:] if root_only else test.labels
        for pred, label in zip(preds.max(1)[1].data.tolist(), labels):
            num_node += 1
            if pred == label:
                accuracy += 1
    accuracy = accuracy / num_node * 100
    return accuracy


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
