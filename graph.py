import  numpy as np

import chainer
from chainer import cuda, Function, gradient_check,\
    Variable, optimizers, serializers, utils

from chainer import  Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


l = L.LSTM(100,50)
l.reset_state()

x = Variable(np.array([1], dtype=np.float32))
y = Variable(np.array([2], dtype=np.float32))
z = Variable(np.array([3], dtype=np.float32))


print(x)

w = (x - 2*y - 1)**2 + (y*z - 1)**2 + 1
print(w.data)

w.backward()
print(x.grad)