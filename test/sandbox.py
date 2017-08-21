import pandas as pd
from src.datamanager import DataManager


train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)

dataset = pd.concat([train,test,dev])


dm = DataManager(batch_size=32)

dataset = dm.init(dataset)

dm.set_max_len(25)




import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import embedding_ops
from copy import deepcopy
from time import time

num_symbols = dm.vocab+dm.start_idx
embedding_size = 200
cell_num = 64
dropout_keep_prob = 0.8
layer_num = 2
epochs = 25





inputs = [tf.placeholder(tf.int32, shape=(None, ), name='inp_token_t%i'%i) for i in range(dm.max_seq_len)]
labels = tf.placeholder(tf.float32, shape=(None, 3), name='labels')


def feed_dict(X, y):
    fd = {inputs[t]: X[t] for t in range(dm.max_seq_len)}
    fd.update({labels: y})
    return fd

cell = rnn.BasicLSTMCell(cell_num)
cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
cells = [deepcopy(cell) for i in range(layer_num)]
cell = rnn.MultiRNNCell(cells)


with tf.variable_scope('encoder'):
    embedding = tf.get_variable("embedding", [num_symbols, embedding_size])
    emb_inputs = [tf.nn.embedding_lookup(embedding, i) for i in inputs]
    enc_output, _ = rnn.static_rnn(cell, emb_inputs, dtype='float32')

with tf.variable_scope('output'):
    output = tf.layers.dense(enc_output[-1], units=3, name='dense')
    pred = tf.nn.softmax(output, name='softmax')

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        start = time()
        print('Epoch %i' % (epoch + 1))
        # Training
        generator = dm.batch_gen(train)
        for _X,_y in generator:
            _, train_loss, train_acc = sess.run([train_op, cross_entropy, accuracy], feed_dict=feed_dict(_X,_y))
            print('\rTain \tloss:%4.8f \tacc:%4.2f%%' % (train_loss, train_acc), end='')

        #Testing
        X_, y_ = dm.input_ready(test, tokenize=True)
        test_loss, test_acc = sess.run([cross_entropy, accuracy], feed_dict=feed_dict(X_,y_))
        print('\nVal \tloss:%4.8f \tacc:%4.2f%%' % (test_loss, test_acc))

        end = time()
        print('\nEpoch time: %is\n' % (end - start))







##################
# Debug
##################
# X_train, y_train = dm.input_ready(train, tokenize=True)
# X_test, y_test = dm.input_ready(test, tokenize=True)
#
# gen = dm.batch_gen(train)
# _X, _y = next(gen)

# sess = tf.InteractiveSession()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# emb_inputs[1].eval(feed_dict(X_train,y_train), sess)