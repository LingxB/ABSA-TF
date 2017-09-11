
import pandas as pd
from src.datamanager import AttDataManager


train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)

dataset = pd.concat([train,test,dev])

dm = AttDataManager(batch_size=25)

dataset = dm.init(dataset)

vocab_in_freq_order = [w for w,c in dm.freq_dist.most_common()]


embedding_path = 'data/glove_840B_300d/glove.840B.300d.txt'
output_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'


d_vec = {}
with open(embedding_path,'r',encoding='utf-8') as f:
    for idx,line in enumerate(f):
        word,vector = line.strip().split(' ', 1)
        d_vec.update({word:vector})


with open(output_path, 'w', encoding='utf-8') as o:
    for word in vocab_in_freq_order:
        if word in d_vec:
            o.write(word + ' ' + d_vec[word] + '\n')
        else:
            print(word)
            o.write(word + '\n')


sem_emb = pd.read_csv(output_path, sep=' ', header=None, index_col=[0])