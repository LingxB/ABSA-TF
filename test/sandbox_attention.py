
import pandas as pd
from src.datamanager import AttDataManager

embedding_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'
train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)

dataset = pd.concat([train,test,dev])


dm = AttDataManager(batch_size=25)

embedding_frame = pd.read_csv(embedding_path, sep=' ', header=None, index_col=[0])

dataset = dm.init(dataset, embedding_frame=embedding_frame)

#X, asp, y = dm.input_ready(train, True)

dm.set_max_len(79)


import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from copy import deepcopy
from time import time

num_symbols = dm.vocab+dm.start_idx
embedding_size = 300 # dw
num_aspects = dm.n_asp
asp_embedding_size = 100 #da
cell_num = 300 # d
dropout_keep_prob = 0.5
layer_num = 2
epochs = 25
seq_len = dm.max_seq_len
epsilon = 0.01
initializer = tf.random_uniform_initializer(-epsilon, epsilon)
beta = 0.001 #l2 param




inputs = [tf.placeholder(tf.int32, shape=(None, ), name='inp_token_t%i'%i) for i in range(dm.max_seq_len)]
asp_inputs = tf.placeholder(tf.int32, shape=(None, ), name='aspect_token')
labels = tf.placeholder(tf.float32, shape=(None, 3), name='labels')


def feed_dict(X, asp, y):
    fd = {inputs[t]: X[t] for t in range(dm.max_seq_len)}
    fd.update({asp_inputs: asp})
    fd.update({labels: y})
    return fd

cell = rnn.BasicLSTMCell(cell_num)
#cell = rnn.LSTMCell(cell_num, initializer=initializer)
#cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
cells = [deepcopy(cell) for i in range(layer_num)]
cell = rnn.MultiRNNCell(cells)

with tf.name_scope('embedding'):
    if dm.use_pretrained_embedding:
        pre_trained_emb = embedding_frame.dropna().values.astype('float32')
        pre_trained_embedding = tf.get_variable(name="pre_trained_embedding", shape=pre_trained_emb.shape,
                                                initializer=tf.constant_initializer(pre_trained_emb), trainable=True)
        pad_embedding = tf.get_variable('pad_embedding', (dm.start_idx, embedding_size), dtype=tf.float32, initializer=initializer)
        embedding = tf.concat([pad_embedding, pre_trained_embedding], axis=0, name='concat_embedding')
    else:
        embedding = tf.get_variable("embedding", (num_symbols, embedding_size), dtype=tf.float32, initializer=initializer)
    emb_inputs = [tf.nn.embedding_lookup(embedding, i) for i in inputs]

    asp_embedding = tf.get_variable('asp_embedding', (num_aspects, asp_embedding_size), dtype=tf.float32, initializer=initializer)
    asp_emb_inputs = tf.nn.embedding_lookup(asp_embedding, asp_inputs)

with tf.name_scope('encoder'):
    enc_output, enc_state = rnn.static_rnn(cell, emb_inputs, dtype='float32')

with tf.variable_scope('attention'):
    H = tf.stack(enc_output, axis=1) #[batch, N, d]
    _H = tf.reshape(tf.stack(enc_output, axis=1), shape=(-1, cell_num)) #[batch*N, d]
    Wh = tf.get_variable('Wh', shape=(cell_num, cell_num), dtype=tf.float32, initializer=initializer) #[d, d]
    Wv = tf.get_variable('Wv', shape=(asp_embedding_size, asp_embedding_size), dtype=tf.float32, initializer=initializer) #[da, da]
    w = tf.get_variable('w', shape=(cell_num+asp_embedding_size, 1), dtype=tf.float32, initializer=initializer) #[d+da, 1]

    WhH = tf.reshape(tf.matmul(_H, Wh), (-1, seq_len, cell_num)) #[batch, N, d]
    Wvva = tf.reshape(tf.matmul(asp_emb_inputs,Wv), (-1, 1, asp_embedding_size)) #[batch, 1, da]
    WvvaeN = tf.tile(Wvva, (1, dm.max_seq_len, 1)) #[batch, N, da]

    M = tf.tanh(tf.concat([WhH,WvvaeN], axis=-1)) #[batch, N, d+da]
    _M = tf.reshape(M, shape=(-1, cell_num+asp_embedding_size)) #[batch*N, d+da]

    alpha = tf.reshape(tf.nn.softmax(tf.matmul(_M,w)), shape=(-1, seq_len, 1)) #[batch, N, 1]

    r = tf.matmul(tf.transpose(H, [0,2,1]), alpha) #[batch, d, 1]

    Wp = tf.get_variable('Wp', shape=(cell_num, cell_num), dtype=tf.float32, initializer=initializer) #[d, d]
    Wx = tf.get_variable('Wx', shape=(cell_num, cell_num), dtype=tf.float32, initializer=initializer) #[d, d]

    _r = tf.reshape(r, (-1, cell_num)) #[batch, d]
    hN = enc_output[-1] #[batch, d] state.h == output[-1]

    h_star = tf.tanh(tf.add(tf.matmul(_r, Wp),tf.matmul(hN, Wx))) #[batch, d]
    h_star = tf.nn.dropout(h_star, dropout_keep_prob)


with tf.name_scope('output'):
    output = tf.layers.dense(h_star, units=3, name='dense')
    pred = tf.nn.softmax(output, name='softmax')

with tf.name_scope('train_ops'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))

    #L2 Regularizer
    reg_params = [param for param in tf.trainable_variables() if not 'embedding:0' in param.name]
    regularizer = tf.add_n([tf.nn.l2_loss(p) for p in reg_params])

    loss = cross_entropy + beta * regularizer

    train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.name_scope('metrics'):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))*100


##################
# Debug
##################
# X_train, asp_train, y_train = dm.input_ready(train, tokenize=True)
# X_test, asp_test, y_test = dm.input_ready(test, tokenize=True)
#
# gen = dm.batch_gen(train)
# _X, _asp, _y = next(gen)
#
# sess = tf.InteractiveSession()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# enc_output[-1].eval(feed_dict(_X,_asp,_y), sess)
# enc_state[-1].h.eval(feed_dict(_X,_asp,_y), sess)
#
#
# embedding.eval(feed_dict(_X,_asp,_y), sess)[:6]
# emb_inputs[0].eval(feed_dict(_X,_asp,_y), sess)
# asp_emb_inputs.eval(feed_dict(_X,_asp,_y), sess)



with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        start = time()
        print('Epoch %i' % (epoch + 1))
        # Training
        generator = dm.batch_gen(train)
        epoch_loss = []
        epoch_acc = []
        for _X,_asp,_y in generator:
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict=feed_dict(_X,_asp,_y))
            epoch_loss.append(train_loss)
            epoch_acc.append(train_acc)
        print('Tain \tloss:%4.8f \tacc:%4.2f%%' % (np.mean(epoch_loss), np.mean(epoch_acc)))

        #Testing
        X_, asp_, y_ = dm.input_ready(test, tokenize=True)
        test_loss, test_acc = sess.run([loss, accuracy], feed_dict=feed_dict(X_,asp_,y_))
        print('Val \tloss:%4.8f \tacc:%4.2f%%' % (test_loss, test_acc))

        end = time()
        print('Epoch time: %is\n' % (end - start))


# Epoch 24
# Tain 	loss:1.15001595 	acc:80.86%
# Val 	loss:1.22192538 	acc:80.37%
# Epoch time: 122s

