
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

model = ATLSTM(dm, embedding_size=300, aspect_embedding_size=100, cell_num=300, layer_num=1, trainable=False, seed=726*104)

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

# **Trainable = True**
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

# **Trainable = False**
# Epoch 1/25
# Tain 	loss:1.30414844 	acc:61.70%
# Val 	loss:1.27511978 	acc:61.93%
# Epoch time: 73s
# Epoch 2/25
# Tain 	loss:1.20999849 	acc:64.32%
# Val 	loss:1.21358609 	acc:62.88%
# Epoch time: 70s
# Epoch 3/25
# Tain 	loss:1.14242935 	acc:67.04%
# Val 	loss:1.10861969 	acc:67.23%
# Epoch time: 73s
# Epoch 4/25
# Tain 	loss:1.10033619 	acc:69.91%
# Val 	loss:1.10692239 	acc:67.99%
# Epoch time: 75s
# Epoch 5/25
# Tain 	loss:1.06994247 	acc:71.58%
# Val 	loss:1.06943262 	acc:70.27%
# Epoch time: 72s
# Epoch 6/25
# Tain 	loss:1.04764605 	acc:72.89%
# Val 	loss:1.04396021 	acc:72.35%
# Epoch time: 73s
# Epoch 7/25
# Tain 	loss:1.03073895 	acc:73.53%
# Val 	loss:1.05348337 	acc:71.59%
# Epoch time: 66s
# Epoch 8/25
# Tain 	loss:1.01303065 	acc:74.47%
# Val 	loss:1.02873290 	acc:72.16%
# Epoch time: 70s
# Epoch 9/25
# Tain 	loss:1.00410056 	acc:74.40%
# Val 	loss:1.04548872 	acc:72.73%
# Epoch time: 53s
# Epoch 10/25
# Tain 	loss:0.98539060 	acc:75.52%
# Val 	loss:1.01037180 	acc:72.35%
# Epoch time: 49s
# Epoch 11/25
# Tain 	loss:0.97539955 	acc:75.50%
# Val 	loss:1.01852953 	acc:72.73%
# Epoch time: 50s
# Epoch 12/25
# Tain 	loss:0.96209389 	acc:75.61%
# Val 	loss:1.01759839 	acc:72.16%
# Epoch time: 50s
# Epoch 13/25
# Tain 	loss:0.95111847 	acc:76.07%
# Val 	loss:0.99396014 	acc:71.97%
# Epoch time: 49s
# Epoch 14/25
# Tain 	loss:0.94154173 	acc:76.38%
# Val 	loss:1.00140095 	acc:72.92%
# Epoch time: 49s
# Epoch 15/25
# Tain 	loss:0.93234867 	acc:76.53%
# Val 	loss:0.99375677 	acc:73.86%
# Epoch time: 48s
# Epoch 16/25
# Tain 	loss:0.92585552 	acc:76.32%
# Val 	loss:0.98094702 	acc:72.92%
# Epoch time: 47s
# Epoch 17/25
# Tain 	loss:0.91453832 	acc:77.33%
# Val 	loss:0.98515534 	acc:72.54%
# Epoch time: 46s
# Epoch 18/25
# Tain 	loss:0.90541703 	acc:77.47%
# Val 	loss:0.97396755 	acc:72.92%
# Epoch time: 46s
# Epoch 19/25
# Tain 	loss:0.89725214 	acc:77.48%
# Val 	loss:0.97023869 	acc:72.16%
# Epoch time: 46s
# Epoch 20/25
# Tain 	loss:0.88328540 	acc:77.99%
# Val 	loss:0.97458839 	acc:73.86%
# Epoch time: 47s
# Epoch 21/25
# Tain 	loss:0.87975860 	acc:78.21%
# Val 	loss:0.98558789 	acc:74.62%
# Epoch time: 47s
# Epoch 22/25
# Tain 	loss:0.86942965 	acc:78.27%
# Val 	loss:0.97063738 	acc:72.35%
# Epoch time: 46s
# Epoch 23/25
# Tain 	loss:0.86492103 	acc:78.44%
# Val 	loss:0.96943516 	acc:73.67%
# Epoch time: 47s
# Epoch 24/25
# Tain 	loss:0.86024052 	acc:78.80%
# Val 	loss:0.96429104 	acc:73.67%
# Epoch time: 47s
# Epoch 25/25
# Tain 	loss:0.85301620 	acc:78.68%
# Val 	loss:0.97546113 	acc:74.05%
# Epoch time: 48s
# INFO:tensorflow:Restoring parameters from ./models/Wed_Sep_27_18-11-04_2017/ATLSTM
# Test 	loss:0.84097284 	acc:80.27%



