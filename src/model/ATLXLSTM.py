from src.model.ATLSTM import ATLSTM
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from time import time





class ATLXLSTM(ATLSTM):

    def __init__(self, lx_embedding_size, lx_emb_initializer='random', **kwargs):
        ATLSTM.__init__(self, **kwargs)
        self.model_name = 'ATLXLSTM'
        self.lx_embedding_size = lx_embedding_size
        self.lx_embedding_initilaizer = lx_emb_initializer
        self.num_polarities = len(self.dm.lx_idx_code)


    def _embedding_with_lx(self, X, asp, lx):
        with tf.variable_scope('embedding'):
            # Word embedding
            if self.use_pretrained_embedding:
                pre_trained_embedding = tf.get_variable(name="pre_trained_embedding", shape=self.embedding_values.shape,
                                                        initializer=tf.constant_initializer(self.embedding_values),
                                                        trainable=self.trainable)
                pad_embedding = tf.get_variable('pad_embedding', (self.dm.start_idx, self.embedding_size), dtype=tf.float32,
                                                initializer=self.initializer)
                embedding = tf.concat([pad_embedding, pre_trained_embedding], axis=0, name='concat_embedding')
            else:
                embedding = tf.get_variable("embedding", (self.num_symbols, self.embedding_size), dtype=tf.float32,
                                            initializer=self.initializer)

            emb_inputs = [tf.nn.embedding_lookup(embedding, i) for i in X]
            # Aspect embedding
            asp_embedding = tf.get_variable('asp_embedding', (self.num_aspects, self.asp_embedding_size), dtype=tf.float32,
                                            initializer=self.initializer)
            asp_emb_inputs = tf.nn.embedding_lookup(asp_embedding, asp)
            # Lexcion embedding
            if self.lx_embedding_initilaizer == 'random':
                lx_init = self.initializer
                trainable = True
            elif self.lx_embedding_initilaizer == 'fixed':
                values = np.array([[0]*self.lx_embedding_size,[-1]*self.lx_embedding_size,[1]*self.lx_embedding_size], dtype='float32')
                lx_init = tf.constant_initializer(values)
                trainable = False
            else:
                raise KeyError('lx_emb_initializer %s not implemented.'%self.lx_embedding_initilaizer)

            lx_embedding = tf.get_variable('lx_embedding', (self.num_polarities, self.lx_embedding_size), dtype=tf.float32,
                                           initializer=lx_init,
                                           trainable=trainable)
            lx_emb_inputs = [tf.nn.embedding_lookup(lx_embedding, i) for i in lx]

            return emb_inputs, asp_emb_inputs, lx_emb_inputs


    def _attention_with_lx(self, enc_outputs, asp_emb_inputs, lx_emb_inputs):
        # Reshape inputs
        with tf.variable_scope('attention'):
            H = tf.stack(enc_outputs, axis=1)  # [batch, N, d]
            _H = tf.reshape(tf.stack(enc_outputs, axis=1), shape=(-1, self.cell_num))  # [batch*N, d]

            L = tf.stack(lx_emb_inputs, axis=1) # [batch, N, dl]
            _L = tf.reshape(L, shape=(-1, self.lx_embedding_size)) # [batch*N, dl]

            # Attention variables
            Wh = tf.get_variable('Wh', shape=(self.cell_num, self.cell_num), dtype=tf.float32,
                                 initializer=self.initializer)  # [d, d]
            Wv = tf.get_variable('Wv', shape=(self.asp_embedding_size, self.asp_embedding_size), dtype=tf.float32,
                                 initializer=self.initializer)  # [da, da]
            # add lexicon embedding projection variable
            Wl = tf.get_variable('Wl', shape=(self.lx_embedding_size, self.lx_embedding_size), dtype=tf.float32,
                                 initializer=self.initializer) # [dl, dl]

            w = tf.get_variable('w', shape=(self.cell_num + self.lx_embedding_size + self.asp_embedding_size, 1), dtype=tf.float32,
                                initializer=self.initializer)  # [d+da, 1]
            Wp = tf.get_variable('Wp', shape=(self.cell_num, self.cell_num), dtype=tf.float32,
                                 initializer=self.initializer)  # [d, d]
            Wx = tf.get_variable('Wx', shape=(self.cell_num, self.cell_num), dtype=tf.float32,
                                 initializer=self.initializer)  # [d, d]

            # Attention operations
            WhH = tf.reshape(tf.matmul(_H, Wh), (-1, self.seq_len, self.cell_num))  # [batch, N, d]
            Wvva = tf.reshape(tf.matmul(asp_emb_inputs, Wv), (-1, 1, self.asp_embedding_size))  # [batch, 1, da]
            WvvaeN = tf.tile(Wvva, (1, self.seq_len, 1))  # [batch, N, da]
            # add lexicon embedding projection
            WlL = tf.reshape(tf.matmul(_L, Wl), (-1, self.seq_len, self.lx_embedding_size)) #[batch, N, dl]

            M = tf.tanh(tf.concat([WhH, WlL, WvvaeN], axis=-1))  # [batch, N, d+dl+da]
            _M = tf.reshape(M, shape=(-1, self.cell_num + self.lx_embedding_size + self.asp_embedding_size))  # [batch*N, d+dl+da]

            alpha = tf.reshape(tf.nn.softmax(tf.matmul(_M, w)), shape=(-1, self.seq_len, 1))  # [batch, N, 1]

            r = tf.matmul(tf.transpose(H, [0, 2, 1]), alpha)  # [batch, d, 1]

            _r = tf.reshape(r, (-1, self.cell_num))  # [batch, d]
            hN = enc_outputs[-1]  # [batch, d] state.h == output[-1]

            h_star = tf.tanh(tf.add(tf.matmul(_r, Wp), tf.matmul(hN, Wx)))  # [batch, d]

            h_star = tf.nn.dropout(h_star, self.dropout_keep_prob)

        return h_star

    def _feed_dict_with_lx(self, X, asp, lx, y):
        fd = {self.inputs[t]: X[t] for t in range(self.seq_len)}
        fd.update({self.asp_inputs: asp})
        fd.update({self.lx_inputs[t]: lx[t] for t in range(self.seq_len)})
        fd.update({self.labels: y})
        return fd

    def _create_graph_with_lx(self):

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Placeholders
            with tf.name_scope('inputs'):
                self.inputs = [tf.placeholder(tf.int32, shape=(None,), name='inp_token_t%i' % i)
                               for i in range(self.seq_len)]

                self.asp_inputs = tf.placeholder(tf.int32, shape=(None,), name='aspect_token')

                self.lx_inputs = [tf.placeholder(tf.int32, shape=(None,), name='inp_token_pol_t%i' % i)
                                  for i in range(self.seq_len)]

                self.labels = tf.placeholder(tf.float32, shape=(None, 3), name='labels')

            # Cell
            cell = self._create_cell()

            # Embedding
            emb_inputs, asp_emb_inputs, lx_emb_inputs = self._embedding_with_lx(self.inputs, self.asp_inputs, self.lx_inputs)

            # LSTM encoder
            enc_outputs, enc_state = rnn.static_rnn(cell, emb_inputs, dtype='float32')

            # Attention
            h_star = self._attention_with_lx(enc_outputs, asp_emb_inputs, lx_emb_inputs)

            # Output layer
            with tf.name_scope('output'):
                output = tf.layers.dense(h_star, units=3, name='dense', kernel_initializer=self.initializer)
                self.pred = tf.nn.softmax(output, name='softmax')

            # Train ops
            with tf.name_scope('train_ops'):
                cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=output))

                # L2 Regularizer
                reg_params = [param for param in tf.trainable_variables() if not 'embedding' in param.name]
                # print(reg_params)
                regularizer = tf.add_n([tf.nn.l2_loss(p) for p in reg_params])

                self.loss = cross_entropy + self.l2_param * regularizer
                tf.summary.scalar('loss', self.loss)

                self.train_op = self.optimizer.minimize(self.loss)

            with tf.name_scope('metrics'):
                correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
                tf.summary.scalar('accuracy', self.accuracy)

            summary_weights = [w for w in tf.trainable_variables() if 'embedding' not in w.name]
            for w in summary_weights:
                tf.summary.histogram(w.name.strip(':0'), w)
            self.summary_op = tf.summary.merge_all()

    def train(self, train_data, epochs, val_data=None, verbose=1, **kwargs):
        self.optimizer = kwargs.get('optimizer', tf.train.AdagradOptimizer(0.01))
        # Create graph
        if self.graph is None:
            self._create_graph_with_lx()
        # Create save ckpt path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        # Write word embedding tsv
        self.dm.write_embedding_tsv(self.model_path) # TODO: Add polarity tsv

        with tf.Session(graph=self.graph) as sess:
            # Create train/val writer
            train_writer = tf.summary.FileWriter(self.model_path + 'train', sess.graph)
            val_writer = tf.summary.FileWriter(self.model_path + 'val')

            init = tf.global_variables_initializer()
            sess.run(init)
            self.saver = tf.train.Saver()

            for epoch in range(epochs):
                start = time()
                print('Epoch %i/%i' % (epoch + 1, epochs))
                # Training
                generator = self.dm.batch_gen(train_data)
                epoch_loss = []
                epoch_acc = []
                for _X, _asp, _lx, _y in generator:
                    _, train_summary, train_loss, train_acc = sess.run([self.train_op, self.summary_op, self.loss, self.accuracy],
                                                                       feed_dict=self._feed_dict_with_lx(_X, _asp, _lx, _y))
                    epoch_loss.append(train_loss)
                    epoch_acc.append(train_acc)
                    train_writer.add_summary(train_summary, epoch * self.dm.n_batchs + 1)
                    if verbose == 2:
                        print('\rTain \tloss:%4.8f \tacc:%4.2f%%' % (train_loss, train_acc), end='')
                if verbose == 2:
                    print('\n')

                if verbose == 1:
                    print('Tain \tloss:%4.8f \tacc:%4.2f%%' % (np.mean(epoch_loss), np.mean(epoch_acc)))

                # Testing
                if val_data is not None:
                    X_, asp_, lx_, y_ = self.dm.input_ready(val_data, tokenize=True)
                    val_summary, test_loss, test_acc = sess.run([self.summary_op, self.loss, self.accuracy],
                                                                feed_dict=self._feed_dict_with_lx(X_, asp_, lx_, y_))
                    val_writer.add_summary(val_summary, epoch * self.dm.n_batchs + 1)
                    print('Val \tloss:%4.8f \tacc:%4.2f%%' % (test_loss, test_acc))

                end = time()
                print('Epoch time: %is\n' % (end - start))

            self.saver.save(sess, self.model_path + self.model_name)


    def predict(self, test_data, verbose=1):
        with tf.Session(graph=self.graph) as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            X_test, asp_test, lx_test, y_test = self.dm.input_ready(test_data, tokenize=True)
            test_pred, test_loss, test_acc = sess.run([self.pred, self.loss, self.accuracy],
                                                      feed_dict=self._feed_dict_with_lx(X_test, asp_test, lx_test, y_test))
            if verbose == 1:
                print('Test \tloss:%4.8f \tacc:%4.2f%%' % (test_loss, test_acc))
            return test_pred



