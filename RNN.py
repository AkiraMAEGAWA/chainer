import numpy as np
import chainer
from chainer import  cuda, Function, gradient_check, \
    Variable, optimizers, serializers, utils
from chainer import  Link, Chain, ChainList
import  chainer.functions as F
import chainer.links as L


#in ptb.train.txt not exist '<eos>' but exist '<unk>'
vocab = {}
def load_data(filename):
    global vocab
    words = open(filename).read().replace('\n', '<unk>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]

    return dataset

train_data = load_data('ptb.train.txt')
unk_id = vocab['<unk>']
print(unk_id)

class  MyRNN(chainer.Chain):
    def __init__(self, v, k):
        super(MyRNN, self).__init__()