from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
from pandas import get_dummies
from src.utils import freq_dist, w_index

class DataManager(object):

    def __init__(self, batch_size, max_seq_len=None, sentcol='SENT', clscol='CLS', start_idx=3):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.sentcol = sentcol
        self.clscol = clscol
        self.start_idx = start_idx
        self.ready = None

    def set_max_len(self, maxlen):
        self.max_seq_len = maxlen

    def tokenize(self, df):
        _df = df.copy()
        _df['TOKENS'] = _df[self.sentcol].str.split()
        _df['TLEN'] = _df['TOKENS'].apply(lambda x: len(x))
        return _df

    def init(self, dataset):
        """Must initialize with complete dataset"""
        _df = self.tokenize(dataset)
        self.freq_dist = freq_dist(_df['TOKENS'])
        self.w_idx = w_index(self.freq_dist, self.start_idx)
        self.vocab = len(self.w_idx)
        print('Longest sent length: %i' %_df['TLEN'].max())
        _df['TLEN'].plot('hist')
        return _df

    def input_ready(self, df, tokenize=False):
        if tokenize:
            _df = self.tokenize(df)
        else:
            _df = df.copy()
        X = _df['TOKENS'].apply(lambda x: [self.w_idx[w] for w in x]).values # Use w_idx to get word index
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
            if i == self.n_batchs - 1:
                _X = [X[t][current:] for t in range(self.max_seq_len)]
                _y = y[current:, :]
            else:
                _X = [X[t][current:self.batch_size * (i + 1)] for t in range(self.max_seq_len)]
                _y = y[current:self.batch_size * (i + 1), :]
            current = self.batch_size * (i + 1)
            yield _X, _y
