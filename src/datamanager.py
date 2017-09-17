from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
from pandas import get_dummies
from src.utils import freq_dist, w_index
from pandas import Series
from numpy import int32

class DataManager(object):

    def __init__(self, batch_size, max_seq_len=None, aspcol='ASP', sentcol='SENT', clscol='CLS', start_idx=3):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.aspcol = aspcol
        self.sentcol = sentcol
        self.clscol = clscol
        self.start_idx = start_idx
        self.ready = None
        assert self.start_idx > 2, 'Start index > 2 for padding with 0 and rest for out of vocabulary words.'

    def set_max_len(self, maxlen):
        self.max_seq_len = maxlen

    def tokenize(self, df):
        _df = df.copy()
        _df['TOKENS'] = _df[self.sentcol].str.split()
        _df['TLEN'] = _df['TOKENS'].apply(lambda x: len(x))
        return _df

    def init(self, dataset, embedding_frame=None, lexicon_frame=None, **kwargs):
        """
        Must initialize with complete dataset.
        embedding_frame is a dataframe with word as index, word vector as data. Out of vocabulary word will be dropped
        if exist. In the embedding_frame, word index is ordered as freq distribution in the corpus.
        """
        _df = self.tokenize(dataset)
        if lexicon_frame is not None:
            self.lx_idx_code =  kwargs.get('lx_idx_code', {-1: 1, 0: 0, 1: 2})# 0, -1, 1 corresponds to embedding matrix row idx [0,1,2]
            assert self.lx_idx_code[0]==0, 'Neutral word / OOV word / Padding must share same symbol: 0'
            self.lx_idx = {w: self.lx_idx_code.get(v.values[0], 0) for w, v in lexicon_frame.iterrows()}
        if embedding_frame is not None:
            self.use_pretrained_embedding = True
            self.embedding_words = embedding_frame.dropna().index.values.tolist()
            self.w_idx = {w:i+self.start_idx for i,w in enumerate(self.embedding_words)}
            self.pretrained_embedding_values = embedding_frame.dropna().values.astype('float32')
        else:
            self.freq_dist = freq_dist(_df['TOKENS'])
            self.embedding_words = [w for w,c in self.freq_dist.most_common()]
            self.w_idx = w_index(self.freq_dist, self.start_idx) # Word index starts from self.start_idx
        self.aspect_words = sorted(_df[self.aspcol].unique())
        self.asp_idx = {w:i for i,w in enumerate(self.aspect_words)} # Aspect idx starts from 0
        self.vocab = len(self.w_idx)
        self.n_asp = _df[self.aspcol].unique().shape[0]
        self.n_classes = _df[self.clscol].astype(str).unique().shape[0]
        print('Longest sent length: %i' %_df['TLEN'].max())
        _df['TLEN'].plot('hist')
        return _df

    def input_ready(self, df, tokenize=False):
        if tokenize:
            _df = self.tokenize(df)
        else:
            _df = df.copy()
        X = _df['TOKENS'].apply(lambda x: [self.w_idx[w] if w in self.w_idx else 1 for w in x]).values # Use w_idx to get word index
        X = pad_sequences(X, maxlen=self.max_seq_len, padding='post', truncating='post') # Pad sequences
        X = [X[:, i].astype('int32') for i in range(X.shape[1])] # Reshape data into [[batch,feats_t1], [batch,feats_t2], ...]
        y = get_dummies(_df[self.clscol].astype(str)).values.astype('float32') # Create one-hot encoded labels
        return X, y

    def batch_gen(self, df):
        if self.ready is None:
            self.ready = self.input_ready(df, tokenize=True)
        X, y = self.ready
        size = X[0].shape[0]
        self.n_batchs = size// self.batch_size if size % self.batch_size == 0 else size // self.batch_size + 1
        current = 0
        for i in range(self.n_batchs):
            next_batch = self.batch_size * (i + 1)
            if i == self.n_batchs - 1:
                _X = [X[t][current:] for t in range(self.max_seq_len)]
                _y = y[current:, :]
            else:
                _X = [X[t][current:next_batch] for t in range(self.max_seq_len)]
                _y = y[current:next_batch, :]
            current = next_batch
            yield _X, _y

    def write_embedding_tsv(self, path):
        with open(path+'word_embedding.tsv', 'w', encoding='utf-8') as f:
            for word in self.embedding_words:
                f.write(word+'\n')
        with open(path+'aspect_embedding.tsv', 'w', encoding='utf-8') as f:
            for aspect in self.aspect_words:
                f.write(aspect+'\n')


class AttDataManager(DataManager):

    def __init__(self, **kwargs):
        DataManager.__init__(self, **kwargs)


    def input_ready(self, df, tokenize=False):
        if tokenize:
            _df = self.tokenize(df)
        else:
            _df = df.copy()
        X = _df['TOKENS'].apply(lambda x: [self.w_idx[w] if w in self.w_idx else 1 for w in x]).values
        X = pad_sequences(X, maxlen=self.max_seq_len, padding='post', truncating='post')
        X = [X[:, i].astype('int32') for i in range(X.shape[1])]

        asp = _df[self.aspcol].apply(lambda w: int32(self.asp_idx[w])).values.tolist()

        y = get_dummies(_df[self.clscol].astype(str)).values.astype('float32')

        return X, asp, y

    def batch_gen(self, df):
        if self.ready is None:
            self.ready = self.input_ready(df, tokenize=True)
        X, asp, y = self.ready
        size = X[0].shape[0]
        self.n_batchs = size// self.batch_size if size % self.batch_size == 0 else size // self.batch_size + 1
        current = 0
        for i in range(self.n_batchs):
            next_batch = self.batch_size * (i + 1)
            if i == self.n_batchs - 1:
                _X = [X[t][current:] for t in range(self.max_seq_len)]
                _asp = asp[current:]
                _y = y[current:, :]
            else:
                _X = [X[t][current:next_batch] for t in range(self.max_seq_len)]
                _asp = asp[current:next_batch]
                _y = y[current:next_batch, :]
            current = next_batch
            yield _X, _asp, _y


class ATLXDataManager(DataManager):

    def __init__(self, **kwargs):
        DataManager.__init__(self, **kwargs)

    def input_ready(self, df, tokenize=False):
        if tokenize:
            _df = self.tokenize(df)
        else:
            _df = df.copy()
        X = _df['TOKENS'].apply(lambda x: [self.w_idx.get(w, 1) for w in x]).values
        X = pad_sequences(X, maxlen=self.max_seq_len, padding='post', truncating='post')
        X = [X[:, i].astype('int32') for i in range(X.shape[1])]

        asp = _df[self.aspcol].apply(lambda w: int32(self.asp_idx[w])).values.tolist()

        lx = _df['TOKENS'].apply(lambda x: [self.lx_idx.get(w, 0) for w in x]).values
        lx = pad_sequences(lx, maxlen=self.max_seq_len, padding='post', truncating='post')
        lx = [lx[:, i].astype('int32') for i in range(lx.shape[1])]

        y = get_dummies(_df[self.clscol].astype(str)).values.astype('float32')

        return X, asp, lx, y

    def batch_gen(self, df):
        if self.ready is None:
            self.ready = self.input_ready(df, tokenize=True)
        X, asp, lx, y = self.ready
        size = X[0].shape[0]
        self.n_batchs = size// self.batch_size if size % self.batch_size == 0 else size // self.batch_size + 1
        current = 0
        for i in range(self.n_batchs):
            next_batch = self.batch_size * (i + 1)
            if i == self.n_batchs - 1:
                _X = [X[t][current:] for t in range(self.max_seq_len)]
                _asp = asp[current:]
                _lx = [lx[t][current:] for t in range(self.max_seq_len)]
                _y = y[current:, :]
            else:
                _X = [X[t][current:next_batch] for t in range(self.max_seq_len)]
                _asp = asp[current:next_batch]
                _lx = [lx[t][current:next_batch] for t in range(self.max_seq_len)]
                _y = y[current:next_batch, :]
            current = next_batch
            yield _X, _asp, _lx, _y









# class BucketAttDataManager(DataManager):
#     # TODO: Finish the dynamic implementation
#
#     def __init__(self, **kwargs):
#         DataManager.__init__(self, **kwargs)
#
#
#     def input_ready(self, df, tokenize=False):
#         if tokenize:
#             _df = self.tokenize(df)
#         else:
#             _df = df.copy()
#
#         X = _df['TOKENS'].apply(lambda x: [self.w_idx[w] if w in self.w_idx else 1 for w in x]).values.tolist()
#
#         xlen = _df['TLEN'].values.astype('int32').tolist()
#
#         asp = _df[self.aspcol].apply(lambda w: int32(self.asp_idx[w])).values.tolist()
#
#         y = get_dummies(_df[self.clscol].astype(str)).values.astype('float32')
#
#         return X, xlen, asp, y
#
#     def batch_gen(self, df):
#         if self.ready is None:
#             self.ready = self.input_ready(df, tokenize=True)
#         X, xlen, asp, y = self.ready
#         size = len(X)
#         self.n_batchs = size// self.batch_size if size % self.batch_size == 0 else size // self.batch_size + 1
#         current = 0
#         for i in range(self.n_batchs):
#             next_batch = self.batch_size * (i + 1)
#             if i == self.n_batchs - 1:
#                 _X = X[current,:]
#                 _xlen = xlen[current,:]
#                 _asp = asp[current:]
#                 _y = y[current:, :]
#             else:
#                 _X = X[current:next_batch]
#                 _xlen = xlen[current:next_batch]
#                 _asp = asp[current:next_batch]
#                 _y = y[current:next_batch, :]
#             current = next_batch
#             yield _X, _xlen, _asp, _y
