
import pandas as pd
from src.datamanager import ATLXDataManager
from src.model.ATLXLSTM import ATLXLSTM

lexicon_path = 'data/Lexicon/lexicon_sample500.csv'
test_lexicon_path = 'data/Lexicon/lexicon_sample500+++.csv'
embedding_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'
train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)
embedding_frame = pd.read_csv(embedding_path, sep=' ', header=None, index_col=[0])
lexicon_frame = pd.read_csv(lexicon_path, index_col=[0])
test_lexicon_frame = pd.read_csv(test_lexicon_path, index_col=[0])
dataset = pd.concat([train,test,dev])

dm = ATLXDataManager(batch_size=25)

dataset = dm.init(dataset,
                  embedding_frame=embedding_frame,
                  lexicon_frame=lexicon_frame,
                  test_lexicon_frame=test_lexicon_frame)


dm.set_max_len(79)



model = ATLXLSTM(datamanager=dm,
                 embedding_size=300,
                 aspect_embedding_size=100,
                 lx_embedding_size=5,
                 lx_emb_initializer='fixed',
                 cell_num=300,
                 layer_num=1,
                 trainable=True,
                 seed=726*104)

model.train(train_data=train, epochs=25, val_data=dev)


pred = model.predict(test_data=test)
pred_2nd = model.predict(test_data=test, use_second_lexicon=True)

test['PRED'] = pred.argmax(axis=1)-1
test = test.join(pd.DataFrame(pred, columns=['P-1','P0','P+1']))
test['T/F'] = test.CLS==test.PRED
test['PRED_2nd'] = pred_2nd.argmax(axis=1)-1
test = test.join(pd.DataFrame(pred_2nd, columns=['P2nd-1','P2nd0','P2nd+1']))
test['T/F_2nd'] = test.CLS==test['PRED_2nd']


test.to_csv(model.model_path+'test_results.csv', index=False)

