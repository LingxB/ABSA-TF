from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
from pandas import get_dummies
from src.utils import freq_dist, w_index

class DataManager(object):

    def __init__(self, max_seq_len=None, sentcol='SENT', clscol='CLS', start_idx=3):
        self.max_seq_len = max_seq_len
        self.sentcol = sentcol
        self.clscol = clscol
        self.start_idx = start_idx

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
        #X = [X[:, i].reshape(-1, 1).astype('int32') for i in range(X.shape[1])] # Reshape data into [[batch,feats_t1], [batch,feats_t2], ...]

        y = get_dummies(_df[self.clscol].astype(str)).values.astype('float32') # Create one-hot encoded labels

        return X, y









def global_stats(df):
    df['TOKENS'] = df.SENT.apply(lambda x: x.split())
    df['TLEN'] = df.TOKENS.apply(lambda x: len(x))
    #df['CLASS'] = ~df.POLARITY.str.contains('Negative')
    c = freq_dist(df.TOKENS)
    w_idx = w_index(c, start_idx=1)
    config.update({'vocab':len(w_idx)})
    return w_idx

def input_ready(df, mlen, w_idx):
    df['TOKENS'] = df.SENT.apply(lambda x: x.split())
    df['TLEN'] = df.TOKENS.apply(lambda x: len(x))
    #df['CLASS'] = ~df.POLARITY.str.contains('Negative')

    data = df2feats(df, 'TOKENS', w_idx)
    X = sequence.pad_sequences(data, maxlen=mlen).astype('float32')
    y = pd.get_dummies(df.CLS).values.astype('float32')
    return X,y