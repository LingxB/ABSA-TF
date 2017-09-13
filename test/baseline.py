
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

model = ATLSTM(dm, embedding_size=300, aspect_embedding_size=100, cell_num=300, layer_num=1, trainable=True)

model.train(train_data=train, epochs=25, val_data=dev)

pred = model.predict(test_data=test)


# Epoch 1/25
# Tain 	loss:1.29749370 	acc:61.81%
# Val 	loss:1.24137735 	acc:63.26%
# Epoch time: 57s
# Epoch 2/25
# Tain 	loss:1.20134890 	acc:64.52%
# Val 	loss:1.16568875 	acc:65.34%
# Epoch time: 60s
# Epoch 3/25
# Tain 	loss:1.13725114 	acc:68.12%
# Val 	loss:1.11813688 	acc:67.99%
# Epoch time: 56s
# Epoch 4/25
# Tain 	loss:1.09266436 	acc:70.46%
# Val 	loss:1.09833622 	acc:69.70%
# Epoch time: 51s
# Epoch 5/25
# Tain 	loss:1.06022823 	acc:72.06%
# Val 	loss:1.09146309 	acc:67.80%
# Epoch time: 54s
# Epoch 6/25
# Tain 	loss:1.03937733 	acc:72.69%
# Val 	loss:1.06998539 	acc:70.83%
# Epoch time: 54s
# Epoch 7/25
# Tain 	loss:1.01568520 	acc:73.82%
# Val 	loss:1.05001760 	acc:70.83%
# Epoch time: 58s
# Epoch 8/25
# Tain 	loss:0.99688923 	acc:74.26%
# Val 	loss:1.05945873 	acc:70.45%
# Epoch time: 53s
# Epoch 9/25
# Tain 	loss:0.98685485 	acc:74.73%
# Val 	loss:1.03461540 	acc:72.35%
# Epoch time: 51s
# Epoch 10/25
# Tain 	loss:0.96651077 	acc:75.83%
# Val 	loss:1.03350449 	acc:71.59%
# Epoch time: 50s
# Epoch 11/25
# Tain 	loss:0.95216125 	acc:76.30%
# Val 	loss:1.03088927 	acc:72.35%
# Epoch time: 53s
# Epoch 12/25
# Tain 	loss:0.94261539 	acc:76.22%
# Val 	loss:1.01833510 	acc:72.92%
# Epoch time: 57s
# Epoch 13/25
# Tain 	loss:0.93051994 	acc:77.01%
# Val 	loss:1.00766337 	acc:72.35%
# Epoch time: 54s
# Epoch 14/25
# Tain 	loss:0.91919571 	acc:77.16%
# Val 	loss:1.00096822 	acc:72.92%
# Epoch time: 53s
# Epoch 15/25
# Tain 	loss:0.90419334 	acc:76.93%
# Val 	loss:0.99675393 	acc:73.11%
# Epoch time: 52s
# Epoch 16/25
# Tain 	loss:0.89095271 	acc:77.77%
# Val 	loss:0.99888563 	acc:73.67%
# Epoch time: 56s
# Epoch 17/25
# Tain 	loss:0.88236147 	acc:78.19%
# Val 	loss:0.99795091 	acc:73.11%
# Epoch time: 56s
# Epoch 18/25
# Tain 	loss:0.87205422 	acc:78.84%
# Val 	loss:0.98841465 	acc:73.11%
# Epoch time: 52s
# Epoch 19/25
# Tain 	loss:0.85843545 	acc:78.63%
# Val 	loss:0.96082723 	acc:73.30%
# Epoch time: 54s
# Epoch 20/25
# Tain 	loss:0.84588349 	acc:79.62%
# Val 	loss:0.95784163 	acc:74.43%
# Epoch time: 53s
# Epoch 21/25
# Tain 	loss:0.83227754 	acc:80.10%
# Val 	loss:0.97854078 	acc:73.67%
# Epoch time: 53s
# Epoch 22/25
# Tain 	loss:0.82429832 	acc:79.79%
# Val 	loss:0.96123517 	acc:73.11%
# Epoch time: 52s
# Epoch 23/25
# Tain 	loss:0.80498999 	acc:81.26%
# Val 	loss:0.97573584 	acc:74.05%
# Epoch time: 53s
# Epoch 24/25
# Tain 	loss:0.79911894 	acc:81.13%
# Val 	loss:0.98495257 	acc:71.78%
# Epoch time: 59s
# Epoch 25/25
# Tain 	loss:0.78553998 	acc:81.52%
# Val 	loss:0.97246909 	acc:73.30%
# Epoch time: 63s
#
# INFO:tensorflow:Restoring parameters from ./models/Wed_Sep_13_10-42-49_2017/ATLSTM
# Test 	loss:0.85638988 	acc:81.40%

