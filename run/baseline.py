
import pandas as pd
from src.datamanager import AttDataManager
from src.model.ATLSTM import ATLSTM


embedding_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'
train_path = 'data/ATAE-LSTM/train.csv'
test_path = 'data/ATAE-LSTM/test.csv'
dev_path = 'data/ATAE-LSTM/dev.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
dev = pd.read_csv(dev_path)

dataset = pd.concat([train,test,dev])

dm = AttDataManager(batch_size=25)

embedding_frame = pd.read_csv(embedding_path, sep=' ', header=None, index_col=[0])

dataset = dm.init(dataset, embedding_frame=embedding_frame)

dm.set_max_len(79)

model = ATLSTM(dm, embedding_size=300, aspect_embedding_size=100, cell_num=300, layer_num=1, trainable=True, seed=726*104)

model.train(train_data=train, epochs=25, val_data=dev, ramdom_shuffle=True)

pred = model.predict(test_data=test)
test['PRED'] = pred.argmax(axis=1)-1
test = test.join(pd.DataFrame(pred, columns=['P-1','P0','P+1']))
test['T/F'] = test.CLS==test.PRED
test.to_csv(model.model_path+'test_results.csv', index=False)

dev_pred = model.predict(test_data=dev)
dev['PRED'] = dev_pred.argmax(axis=1)-1
dev = dev.join(pd.DataFrame(dev_pred, columns=['P-1','P0','P+1']))
dev['T/F'] = dev.CLS==dev.PRED
dev.to_csv(model.model_path+'dev_results.csv', index=False)

train_pred = model.predict(test_data=train)
train['PRED'] = train_pred.argmax(axis=1)-1
train = train.join(pd.DataFrame(train_pred, columns=['P-1','P0','P+1']))
train['T/F'] = train.CLS==train.PRED
train.to_csv(model.model_path+'train_results.csv', index=False)


# Epoch 1/25
# Tain 	loss:1.30411685 	acc:61.70%
# Val 	loss:1.27498603 	acc:61.93%
# Epoch time: 59s
# Epoch 2/25
# Tain 	loss:1.20970798 	acc:64.32%
# Val 	loss:1.21346557 	acc:62.88%
# Epoch time: 66s
# Epoch 3/25
# Tain 	loss:1.14159775 	acc:67.04%
# Val 	loss:1.10788870 	acc:67.23%
# Epoch time: 60s
# Epoch 4/25
# Tain 	loss:1.09870839 	acc:70.04%
# Val 	loss:1.10613751 	acc:68.37%
# Epoch time: 55s
# Epoch 5/25
# Tain 	loss:1.06724441 	acc:71.74%
# Val 	loss:1.06818259 	acc:70.45%
# Epoch time: 56s
# Epoch 6/25
# Tain 	loss:1.04372871 	acc:72.89%
# Val 	loss:1.04111433 	acc:72.92%
# Epoch time: 57s
# Epoch 7/25
# Tain 	loss:1.02528834 	acc:73.71%
# Val 	loss:1.05166137 	acc:71.59%
# Epoch time: 60s
# Epoch 8/25
# Tain 	loss:1.00582957 	acc:74.97%
# Val 	loss:1.02451622 	acc:72.35%
# Epoch time: 59s
# Epoch 9/25
# Tain 	loss:0.99531078 	acc:75.19%
# Val 	loss:1.04185474 	acc:72.54%
# Epoch time: 64s
# Epoch 10/25
# Tain 	loss:0.97464854 	acc:76.04%
# Val 	loss:1.00438631 	acc:72.92%
# Epoch time: 56s
# Epoch 11/25
# Tain 	loss:0.96239716 	acc:76.47%
# Val 	loss:1.01352763 	acc:73.30%
# Epoch time: 57s
# Epoch 12/25
# Tain 	loss:0.94705290 	acc:76.39%
# Val 	loss:1.01165295 	acc:72.54%
# Epoch time: 59s
# Epoch 13/25
# Tain 	loss:0.93403310 	acc:77.00%
# Val 	loss:0.98864424 	acc:72.54%
# Epoch time: 72s
# Epoch 14/25
# Tain 	loss:0.92200172 	acc:77.71%
# Val 	loss:0.99972910 	acc:73.30%
# Epoch time: 64s
# Epoch 15/25
# Tain 	loss:0.91035539 	acc:77.93%
# Val 	loss:0.99221563 	acc:75.00%
# Epoch time: 62s
# Epoch 16/25
# Tain 	loss:0.90126646 	acc:78.08%
# Val 	loss:0.97768521 	acc:73.48%
# Epoch time: 65s
# Epoch 17/25
# Tain 	loss:0.88720709 	acc:78.80%
# Val 	loss:0.98228455 	acc:73.30%
# Epoch time: 64s
# Epoch 18/25
# Tain 	loss:0.87549847 	acc:78.93%
# Val 	loss:0.96830666 	acc:74.24%
# Epoch time: 62s
# Epoch 19/25
# Tain 	loss:0.86339539 	acc:79.68%
# Val 	loss:0.96973026 	acc:72.92%
# Epoch time: 60s
# Epoch 20/25
# Tain 	loss:0.84779900 	acc:79.76%
# Val 	loss:0.97082013 	acc:74.05%
# Epoch time: 62s
# Epoch 21/25
# Tain 	loss:0.83930814 	acc:80.03%
# Val 	loss:0.98881042 	acc:73.86%
# Epoch time: 65s
# Epoch 22/25
# Tain 	loss:0.82656914 	acc:80.73%
# Val 	loss:0.97034085 	acc:74.24%
# Epoch time: 71s
# Epoch 23/25
# Tain 	loss:0.81743324 	acc:81.03%
# Val 	loss:0.97300476 	acc:75.19%
# Epoch time: 68s
# Epoch 24/25
# Tain 	loss:0.81072628 	acc:81.23%
# Val 	loss:0.96082002 	acc:74.24%
# Epoch time: 64s
# Epoch 25/25
# Tain 	loss:0.79766840 	acc:81.48%
# Val 	loss:0.98424745 	acc:74.81%
# Epoch time: 67s
# INFO:tensorflow:Restoring parameters from ./models/Wed_Sep_27_15-53-34_2017/ATLSTM
# Test 	loss:0.86237544 	acc:81.60%


