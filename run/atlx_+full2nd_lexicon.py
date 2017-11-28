
import pandas as pd
from src.datamanager import ATLXDataManager
from src.model.ATLXLSTM import ATLXLSTM

lexicon_path = 'data/Lexicon/full_lexicon.csv'
test_lexicon_path = 'data/Lexicon/full_lexicon+++.csv'
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
                 lx_embedding_size=1,
                 lx_emb_initializer='fixed',
                 cell_num=300,
                 layer_num=1,
                 trainable=True,
                 concat_emblx=True
                 #seed=726*104
                 )

model.train(train_data=train, epochs=30, val_data=dev, ramdom_shuffle=True)


pred = model.predict(test_data=test)
pred_2nd = model.predict(test_data=test, use_second_lexicon=True)

test['PRED'] = pred.argmax(axis=1)-1
test = test.join(pd.DataFrame(pred, columns=['P-1','P0','P+1']))
test['T/F'] = test.CLS==test.PRED
test['PRED_2nd'] = pred_2nd.argmax(axis=1)-1
test = test.join(pd.DataFrame(pred_2nd, columns=['P2nd-1','P2nd0','P2nd+1']))
test['T/F_2nd'] = test.CLS==test['PRED_2nd']


test.to_csv(model.model_path+'test_results.csv', index=False)


# ATLX+ FULL 1d fixed-trainable
# Epoch 1/30
# Tain 	loss:1.29640651 	acc:62.11%
# Val 	loss:1.19674921 	acc:64.20%
# Epoch time: 45s
# Epoch 2/30
# Tain 	loss:1.19663346 	acc:65.21%
# Val 	loss:1.11715400 	acc:66.48%
# Epoch time: 44s
# Epoch 3/30
# Tain 	loss:1.12464654 	acc:68.82%
# Val 	loss:1.07582343 	acc:69.32%
# Epoch time: 44s
# Epoch 4/30
# Tain 	loss:1.07977486 	acc:70.92%
# Val 	loss:1.07193518 	acc:70.08%
# Epoch time: 45s
# Epoch 5/30
# Tain 	loss:1.04419589 	acc:72.29%
# Val 	loss:1.05627227 	acc:71.78%
# Epoch time: 45s
# Epoch 6/30
# Tain 	loss:1.02421081 	acc:74.16%
# Val 	loss:1.05944884 	acc:70.83%
# Epoch time: 44s
# Epoch 7/30
# Tain 	loss:1.00352693 	acc:75.09%
# Val 	loss:1.02943730 	acc:71.21%
# Epoch time: 44s
# Epoch 8/30
# Tain 	loss:0.98507780 	acc:76.36%
# Val 	loss:1.03350329 	acc:71.59%
# Epoch time: 46s
# Epoch 9/30
# Tain 	loss:0.97175330 	acc:76.72%
# Val 	loss:1.03483677 	acc:71.40%
# Epoch time: 46s
# Epoch 10/30
# Tain 	loss:0.95844966 	acc:77.16%
# Val 	loss:1.02141166 	acc:70.83%
# Epoch time: 46s
# Epoch 11/30
# Tain 	loss:0.94633448 	acc:76.91%
# Val 	loss:1.01951027 	acc:72.35%
# Epoch time: 45s
# Epoch 12/30
# Tain 	loss:0.93504727 	acc:77.04%
# Val 	loss:1.00383008 	acc:72.73%
# Epoch time: 45s
# Epoch 13/30
# Tain 	loss:0.92458940 	acc:77.84%
# Val 	loss:1.00657153 	acc:72.73%
# Epoch time: 44s
# Epoch 14/30
# Tain 	loss:0.90725875 	acc:78.08%
# Val 	loss:0.99918461 	acc:71.78%
# Epoch time: 43s
# Epoch 15/30
# Tain 	loss:0.90084243 	acc:78.94%
# Val 	loss:1.00080216 	acc:71.40%
# Epoch time: 44s
# Epoch 16/30
# Tain 	loss:0.88792282 	acc:78.78%
# Val 	loss:0.99205625 	acc:71.97%
# Epoch time: 44s
# Epoch 17/30
# Tain 	loss:0.87941492 	acc:78.92%
# Val 	loss:0.99299812 	acc:71.78%
# Epoch time: 45s
# Epoch 18/30
# Tain 	loss:0.86570102 	acc:79.71%
# Val 	loss:1.00061190 	acc:71.59%
# Epoch time: 45s
# Epoch 19/30
# Tain 	loss:0.85207409 	acc:80.01%
# Val 	loss:1.01762581 	acc:70.27%
# Epoch time: 43s
# Epoch 20/30
# Tain 	loss:0.84003997 	acc:80.41%
# Val 	loss:0.99194741 	acc:71.97%
# Epoch time: 44s
# Epoch 21/30
# Tain 	loss:0.83623970 	acc:81.18%
# Val 	loss:1.00537789 	acc:72.16%
# Epoch time: 44s
# Epoch 22/30
# Tain 	loss:0.81900012 	acc:81.14%
# Val 	loss:1.01894522 	acc:72.16%
# Epoch time: 44s
# Epoch 23/30
# Tain 	loss:0.80850083 	acc:81.21%
# Val 	loss:1.01359367 	acc:72.16%
# Epoch time: 44s
# Epoch 24/30
# Tain 	loss:0.80647749 	acc:81.17%
# Val 	loss:1.03429139 	acc:70.64%
# Epoch time: 44s
# Epoch 25/30
# Tain 	loss:0.78862602 	acc:82.08%
# Val 	loss:1.04324543 	acc:70.45%
# Epoch time: 44s
# Epoch 26/30
# Tain 	loss:0.77686155 	acc:82.20%
# Val 	loss:1.03957295 	acc:71.97%
# Epoch time: 45s
# Epoch 27/30
# Tain 	loss:0.76645148 	acc:82.90%
# Val 	loss:1.01598740 	acc:73.86%
# Epoch time: 45s
# Epoch 28/30
# Tain 	loss:0.75955135 	acc:82.57%
# Val 	loss:1.03732991 	acc:71.21%
# Epoch time: 44s
# Epoch 29/30
# Tain 	loss:0.74855447 	acc:83.17%
# Val 	loss:1.06983352 	acc:72.35%
# Epoch time: 45s
# Epoch 30/30
# Tain 	loss:0.72783369 	acc:84.33%
# Val 	loss:1.06477261 	acc:71.97%
# Epoch time: 43s
# INFO:tensorflow:Restoring parameters from ./models/Mon_Nov_27_18-26-21_2017/ATLXLSTM
# Test 	loss:0.87935102 	acc:80.88%
# INFO:tensorflow:Restoring parameters from ./models/Mon_Nov_27_18-26-21_2017/ATLXLSTM
# Using second lexicon.
# Test 	loss:0.87469494 	acc:81.09%



#df = pd.read_csv('S:\ebao\ABSA-TF/trained_models/nontrainable_baseline_Wed_Sep_27_18-11-04_2017/test_results.csv')