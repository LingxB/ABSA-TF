import tensorflow as tf
from src.model.ATLSTM import ATLSTM




class ATLXLSTM(ATLSTM):

    def __init__(self, lx_embedding_size, **kwargs):
        ATLSTM.__init__(self, **kwargs)
        self.model_name = ''
        self.lx_embedding_size = lx_embedding_size
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
            lx_embedding = tf.get_variable('lx_embedding', (self.num_polarities, self.lx_embedding_size), dtype=tf.float32,
                                           initializer=self.initializer)
            lx_emb_inputs = [tf.nn.embedding_lookup(lx_embedding, i) for i in lx]

            return emb_inputs, asp_emb_inputs, lx_emb_inputs






# TODO: 1. Embed lexicon inputs
# TODO: 2. Merge lx_emb_inps with enc_output, asp_input
# TODO: 3. Add lx embeeding tsv in dm