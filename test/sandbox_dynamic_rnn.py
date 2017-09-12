import pandas as pd
from src.datamanager import DynamicAttDataManager


embedding_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'
train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)

dataset = pd.concat([train,test,dev])


dm = DynamicAttDataManager(batch_size=25)

embedding_frame = pd.read_csv(embedding_path, sep=' ', header=None, index_col=[0])

dataset = dm.init(dataset, embedding_frame=None)


X, xlen, asp, y = dm.input_ready(train, True)


import tensorflow as tf
import numpy as np

inputs = tf.placeholder(tf.int32, shape=(None, None))

embedding = tf.get_variable('embedding', shape=(dm.vocab+dm.start_idx, 100), dtype=tf.float32)
embedding_inputs = tf.nn.embedding_lookup(embedding, inputs)


##################
# Debug
##################
def feed_dict(X, xlen, asp, y):
    fd = {inputs: X}
    #fd.update({asp_inputs: asp})
    #fd.update({labels: y})
    return fd

X_train, xlen_train, asp_train, y_train = dm.input_ready(train, tokenize=True)
X_test, xlen_test, asp_test, y_test = dm.input_ready(test, tokenize=True)

gen = dm.batch_gen(train)
_X, _xlen, _asp, _y = next(gen)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

embedding_inputs.eval(feed_dict(np.array(_X), _xlen, _asp, _y), sess)