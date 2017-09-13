
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



