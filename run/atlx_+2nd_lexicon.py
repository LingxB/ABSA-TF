
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
                 lx_emb_initializer='fixed_trainable',
                 cell_num=300,
                 layer_num=1,
                 trainable=False,
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


# ATLX+ 1d fixed-trainable
# Epoch 1/30
# Tain 	loss:1.29353917 	acc:61.76%
# Val 	loss:1.31700027 	acc:62.12%
# Epoch time: 55s
# Epoch 2/30
# Tain 	loss:1.20201647 	acc:64.19%
# Val 	loss:1.17795348 	acc:64.39%
# Epoch time: 54s
# Epoch 3/30
# Tain 	loss:1.12952495 	acc:68.29%
# Val 	loss:1.10933876 	acc:67.61%
# Epoch time: 55s
# Epoch 4/30
# Tain 	loss:1.08624637 	acc:70.46%
# Val 	loss:1.07114255 	acc:70.83%
# Epoch time: 54s
# Epoch 5/30
# Tain 	loss:1.05746913 	acc:72.20%
# Val 	loss:1.06059909 	acc:70.83%
# Epoch time: 55s
# Epoch 6/30
# Tain 	loss:1.02948940 	acc:73.03%
# Val 	loss:1.05114281 	acc:71.97%
# Epoch time: 54s
# Epoch 7/30
# Tain 	loss:1.01512897 	acc:74.48%
# Val 	loss:1.03380895 	acc:71.40%
# Epoch time: 53s
# Epoch 8/30
# Tain 	loss:0.99440366 	acc:74.79%
# Val 	loss:1.03533614 	acc:71.78%
# Epoch time: 58s
# Epoch 9/30
# Tain 	loss:0.98084104 	acc:75.62%
# Val 	loss:1.03107965 	acc:72.35%
# Epoch time: 83s
# Epoch 10/30
# Tain 	loss:0.96557677 	acc:76.22%
# Val 	loss:1.02704501 	acc:71.02%
# Epoch time: 115s
# Epoch 11/30
# Tain 	loss:0.94995511 	acc:76.59%
# Val 	loss:1.01807702 	acc:71.78%
# Epoch time: 129s
# Epoch 12/30
# Tain 	loss:0.93529600 	acc:76.52%
# Val 	loss:0.99160910 	acc:73.11%
# Epoch time: 129s
# Epoch 13/30
# Tain 	loss:0.92713511 	acc:77.19%
# Val 	loss:0.98949623 	acc:73.86%
# Epoch time: 132s
# Epoch 14/30
# Tain 	loss:0.91046351 	acc:77.69%
# Val 	loss:1.02389085 	acc:73.11%
# Epoch time: 128s
# Epoch 15/30
# Tain 	loss:0.90449154 	acc:77.83%
# Val 	loss:1.02317512 	acc:72.35%
# Epoch time: 129s
# Epoch 16/30
# Tain 	loss:0.89228374 	acc:78.06%
# Val 	loss:0.99812317 	acc:71.02%
# Epoch time: 130s
# Epoch 17/30
# Tain 	loss:0.87674046 	acc:78.64%
# Val 	loss:1.00579321 	acc:72.35%
# Epoch time: 129s
# Epoch 18/30
# Tain 	loss:0.86607170 	acc:79.24%
# Val 	loss:1.01577067 	acc:73.11%
# Epoch time: 125s
# Epoch 19/30
# Tain 	loss:0.85452080 	acc:79.54%
# Val 	loss:1.01306188 	acc:72.73%
# Epoch time: 131s
# Epoch 20/30
# Tain 	loss:0.84158581 	acc:80.08%
# Val 	loss:0.98890543 	acc:73.48%
# Epoch time: 127s
# Epoch 21/30
# Tain 	loss:0.82948554 	acc:80.51%
# Val 	loss:1.02546453 	acc:72.92%
# Epoch time: 130s
# Epoch 22/30
# Tain 	loss:0.82153934 	acc:80.24%
# Val 	loss:1.00891662 	acc:72.35%
# Epoch time: 127s
# Epoch 23/30
# Tain 	loss:0.80946529 	acc:80.62%
# Val 	loss:1.00199509 	acc:73.30%
# Epoch time: 128s
# Epoch 24/30
# Tain 	loss:0.79387730 	acc:81.54%
# Val 	loss:1.01248646 	acc:74.81%
# Epoch time: 128s
# Epoch 25/30
# Tain 	loss:0.78333437 	acc:82.11%
# Val 	loss:1.00883985 	acc:75.76%
# Epoch time: 131s
# Epoch 26/30
# Tain 	loss:0.77723640 	acc:82.21%
# Val 	loss:1.02841640 	acc:75.00%
# Epoch time: 135s
# Epoch 27/30
# Tain 	loss:0.75807405 	acc:83.32%
# Val 	loss:1.02272868 	acc:74.43%
# Epoch time: 138s
# Epoch 28/30
# Tain 	loss:0.74832112 	acc:83.51%
# Val 	loss:1.01513612 	acc:74.62%
# Epoch time: 134s
# Epoch 29/30
# Tain 	loss:0.73285961 	acc:83.88%
# Val 	loss:1.04804945 	acc:75.76%
# Epoch time: 132s
# Epoch 30/30
# Tain 	loss:0.72167808 	acc:84.61%
# Val 	loss:1.02689135 	acc:74.62%
# Epoch time: 128s
#
# INFO:tensorflow:Restoring parameters from ./models/Thu_Sep_28_00-17-43_2017/ATLXLSTM
# Test 	loss:0.87994528 	acc:81.40%
#
# INFO:tensorflow:Restoring parameters from ./models/Thu_Sep_28_00-17-43_2017/ATLXLSTM
# Using second lexicon.
# Test 	loss:0.87524927 	acc:81.71%

df = pd.read_csv('S:\ebao\ABSA-TF/trained_models/nontrainable_baseline_Wed_Sep_27_18-11-04_2017/test_results.csv')