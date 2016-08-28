import numpy as np
import cPickle
import csv
import parser
from rntn_core import RNTNTree


class RNTNModel:

    def __init__(self, vocab_file=None, wvec_dim=None, num_classes=None, l=None):
        ## Building vocabulary index
        with open(vocab_file, 'r') as f:
            reader = csv.reader(f)
            self.word_index_map = dict((row[0], int(row[1])) for row in reader)

        ## Initializing the parameters that will be learnt
        d = wvec_dim
        C = num_classes
        self.params = {}                         # Parameter hash table
        self.params['L'] = np.random.uniform(    # Word vector matrix (dx|vocabulary|)
            low=-0.0001, high=0.0001, size=(
                d,len(self.word_index_map)))
        self.params['V'] = np.random.uniform(    # Tensor (2dx2dxd)
            low=-0.0001, high=0.0001, size=(2*d,2*d,d))
        self.params['W'] = np.random.normal(     # Weight matrix (dx2d)
            loc=0.0, scale=0.01, size=(d,2*d))
        self.params['b'] = np.random.normal(     # Bias (dx1)
            loc=0.0, scale=0.01, size=(d,1))
        self.params['Ws'] = np.random.normal(    # Weights for the softmax classifier (Cxd)
            loc=0.0, scale=0.01, size=(C,d))
        self.params['bs'] = np.random.normal(    # Bias for the softmax classifier (Cx1)
            loc=0.0, scale=0.01, size=(C,1))

        ## Initializing the gradient accumulators of the parameters
        self.params['dL'] = np.zeros(shape=(d,len(self.word_index_map)))
        self.params['dV'] = np.zeros(shape=(2*d, 2*d, d))
        self.params['dW'] = np.zeros(shape=(d, 2*d))
        self.params['db'] = np.zeros(shape=(d, 1))
        self.params['dWs'] = np.zeros(shape=(C, d))
        self.params['dbs'] = np.zeros(shape=(C, 1))

        ## Initializing the historic gradient accumulators for AdaGrad
        self.params['hdL'] = np.zeros(shape=(d,len(self.word_index_map)))
        self.params['hdV'] = np.zeros(shape=(2*d, 2*d, d))
        self.params['hdW'] = np.zeros(shape=(d, 2*d))
        self.params['hdb'] = np.zeros(shape=(d, 1))
        self.params['hdWs'] = np.zeros(shape=(C, d))
        self.params['hdbs'] = np.zeros(shape=(C, 1))

        self.params['C'] = C # Number of classes
        self.params['d'] = d # Dimensionality of the word vectors

    def make_rntn_tree(self, current, node):
        node.label = int(current.label)
        if not current.subtrees:
            node.leaf = True
            node.word = current.word
            node.word_idx = self.word_index_map[current.word]
        else:
            left = current.subtrees[0]
            node.left = self.make_rntn_tree(left, RNTNTree(self.params))
            right = current.subtrees[1]
            node.right = self.make_rntn_tree(right, RNTNTree(self.params))
        return node

    def __get_rntn_trees(self, filename):
        trees = parser.read_trees(filename)
        trees = [self.make_rntn_tree(tree, RNTNTree(self.params)) for tree in trees]
        return trees

    def get_accuracy(self, trees):
        """ Computes the average fine grained accuracy
            for all the trees provided in the list"""

        def compute_accuracy(tree):
            acc = float(np.argmax(tree.probs) == tree.label)
            if tree.leaf:
                return acc, 1.0
            lacc, lc = compute_accuracy(tree.left)
            racc, rc = compute_accuracy(tree.right)
            return acc + lacc + racc, lc + rc + 1.0
        if type(trees) is list:
            accuracy = 0.0
            for tree in trees:
                a, b = compute_accuracy(tree)
                accuracy += a / b
            accuracy /= len(trees)
            return accuracy
        else:
            return compute_accuracy(trees)

    def train(self, train_file=None, step_size=0.001, lamda=0.01,
                epsilon=1e-12, num_epoch=3, batch_size=None):
        trees = self.__get_rntn_trees(train_file)
        if not batch_size:
            batch_size = 1
        for epoch in xrange(num_epoch):
            for i, tree in enumerate(trees):
                tree.forward_propagate(tree)
                tree.back_propagate(tree, lamda=lamda)
                if (i+1) % batch_size == 0:
                    tree.adagrad_update(step_size, epsilon, batch_size)
            if not (len(trees) % batch_size == 0):
                trees[0].adagrad_update(step_size, epsilon, batch_size)
            print 'Epoch', epoch + 1, 'Accuracy:', self.get_accuracy(trees)

    def test(self, test_file=None):
        trees = self.__get_rntn_trees(test_file)
        for tree in trees:
            tree.forward_propagate(tree)
        accuracy = self.get_accuracy(trees)
        print 'Accuracy:', accuracy

    def save(self, save_file):
        with open(save_file, 'wb') as f:
            cPickle.dump(self.params, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def load(self, save_file):
        with open(save_file, 'r') as f:
            self.params = cPickle.load(f)
