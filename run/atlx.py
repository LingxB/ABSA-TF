
import pandas as pd
from src.datamanager import ATLXDataManager
from src.model.ATLSTM import ATLSTM

lexicon_path = 'data/Lexicon/lexicon_sample500.csv'
embedding_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'
train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)
embedding_frame = pd.read_csv(embedding_path, sep=' ', header=None, index_col=[0])
lexicon_frame = pd.read_csv('data/Lexicon/lexicon_sample500.csv', index_col=[0])
dataset = pd.concat([train,test,dev])

dm = ATLXDataManager(batch_size=25)



dataset = dm.init(dataset, embedding_frame=embedding_frame, lexicon_frame=lexicon_frame)


dm.set_max_len(21)


X, asp, lx, y = next(dm.batch_gen(train))
