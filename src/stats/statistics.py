import pandas as pd
from collections import Counter
import numpy as np

# Dataset
lexicon_path = 'data/Lexicon/lexicon_sample500.csv'
test_lexicon_path = 'data/Lexicon/lexicon_sample500+++.csv'
embedding_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'
train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'

train = pd.read_csv(train_path)
train['SET'] = ['train']*train.shape[0]
test = pd.read_csv(test_path)
test['SET'] = ['test']*test.shape[0]
dev = pd.read_csv(dev_path)
dev['SET'] = ['dev']*dev.shape[0]

embedding_frame = pd.read_csv(embedding_path, sep=' ', header=None, index_col=[0])
lexicon_frame = pd.read_csv(lexicon_path, index_col=[0])
test_lexicon_frame = pd.read_csv(test_lexicon_path, index_col=[0])
dataset = pd.concat([train,test,dev])


# Dataset vocabulary
c = Counter()

_ = [c.update(l) for _,l in dataset.SENT.str.split().items()]

dataset_vocab = c.most_common() #5175
print('Total vocab in dataset: %i' %len(dataset_vocab))

# Words with embedding and without
_embedding_words = embedding_frame.dropna().index.tolist()

no_embedding_words = [w for w,c in dataset_vocab if w not in _embedding_words] #271
embedding_words = [w for w,_ in dataset_vocab if w not in no_embedding_words] #4904

print('Embedding words: %i' %len(embedding_words))
print('Out of embedding scope words: %i' %len(no_embedding_words))


# Lexicon words
print('1st lexicon size: %i'%lexicon_frame.shape[0])
print('2nd lexicon size: %i'%test_lexicon_frame.shape[0])


lx_words_1 = lexicon_frame.index.tolist()
lx_words_2 = test_lexicon_frame.index.tolist()

emb_w_not_in_lx_1 = [w for w in embedding_words if w not in lx_words_1]
emb_w_not_in_lx_2 = [w for w in embedding_words if w not in lx_words_2]

print('Words assigned to netural by default using 1st lexicon: %i' %(len(emb_w_not_in_lx_1)+len(no_embedding_words)))
print('Words assigned to netural by default using 2nd lexicon: %i' %(len(emb_w_not_in_lx_2)+len(no_embedding_words)))

# Stats df
df = pd.DataFrame(data=None, index=[w for w,_ in dataset_vocab])
df['EMBEDDING'] = [True if w in embedding_words else False for w in df.index]
df['IN1STLX'] = [True if w in lx_words_1 else False for w in df.index]
df['IN2NDLX'] = [True if w in lx_words_2 else False for w in df.index]


train_words = ' '.join(train.SENT.values).split()
dev_words = ' '.join(dev.SENT.values).split()
test_words = ' '.join(test.SENT.values).split()

in_train = []
in_dev = []
in_test = []
for w in df.index:
    itr, ide, ite = False, False, False
    if w in train_words:
        itr = True
    elif w in dev_words:
        ide = True
    elif w in test_words:
        ite = True
    else:
        print(w)
    in_train.append(itr)
    in_dev.append(ide)
    in_test.append(ite)

df['INTRAIN'] = in_train
df['INDEV'] = in_dev
df['INTEST'] = in_test
df['POLARITY'] = [test_lexicon_frame.loc[w]['POLARITY'] if w in test_lexicon_frame.index else np.nan for w in df.index]

# n_words from dict, check ploarity correct in training set
pd.set_option('display.max_colwidth', -1)
n_words = 50
chosen_words = lexicon_frame.sample(n_words, random_state=42).index.tolist()
gen_cw = (w for w in chosen_words)

cw = next(gen_cw)


print(cw, lexicon_frame.loc[cw]['POLARITY'])
frame = dataset[dataset.SENT.str.contains(cw)]
print(frame)




# CORRECT baseline, WRONG experiment

baseline = pd.read_csv('trained_models/trainable_baseline_Wed_Sep_27_15-53-34_2017/test_results.csv')
experiment = pd.read_csv('models\Thu_Sep_28_00-17-43_2017/test_results.csv')


experiment[(baseline['T/F']==True) & (experiment['T/F']==False)]
baseline[(baseline['T/F']==True) & (experiment['T/F']==False)]


experiment[(baseline['T/F']==False) & (experiment['T/F']==True)]
baseline[(baseline['T/F']==False) & (experiment['T/F']==True)]




