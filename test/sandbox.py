import pandas as pd
from src.datamanager import DataManager


train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)

dataset = pd.concat([train,test,dev])


dm = DataManager()

dataset = dm.init(dataset)

dm.set_max_len(25)

X_train, y_train = dm.input_ready(train, tokenize=True)
X_test, y_test = dm.input_ready(test, tokenize=True)




import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import embedding_ops

num_symbols = dm.vocab+dm.start_idx
embedding_size = 200

embedding = tf.get_variable("embedding", [num_symbols, embedding_size])


inputs = tf.placeholder(tf.int32, shape=(None, 25))
emb_inps = tf.nn.embedding_lookup(embedding, inputs)


#
# inputs = [tf.placeholder(tf.int32, shape=(None, ), name='inp_token_%i'%i) for i in range(dm.max_seq_len)]
# labels = tf.placeholder(tf.float32, shape=(None, 3), name='labels')
#
# with tf.variable_scope('encoder'):
#     embedding = tf.get_variable("embedding", [num_symbols, embedding_size])
#     emb_inputs = [embedding_ops.embedding_lookup(embedding, i) for i in inputs]
#     emb_inputs = [tf.gather(embedding, i) for i in inputs]