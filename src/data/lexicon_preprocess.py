
import pandas as pd

# Read Bing Liu's lexicon
positive_words = pd.read_csv('data/Lexicon/HuAndLiu/positive-words.txt', skiprows=33, encoding='utf-8')
positive_words.columns = ['WORD']
positive_words['POLARITY'] = 1

negative_words = pd.read_csv('data/Lexicon/HuAndLiu/negative-words.txt', skiprows=33, encoding='utf-8')
negative_words.columns = ['WORD']
negative_words['POLARITY'] = -1
# Concat pos/neg words to full dataframe
lexicon = pd.concat([positive_words,negative_words]).set_index('WORD')

# Search for lexicon words exist in corpus vocabulary
embedding_path = 'data/glove_840B_300d/semeval14_glove.840B.300d.txt'
embedding_frame = pd.read_csv(embedding_path, sep=' ', header=None, index_col=[0])
corpus_vocab = embedding_frame.dropna().index.values
_in_lexicon = [True if word in lexicon.index.tolist() else False for word in corpus_vocab]
in_lexicon = corpus_vocab[_in_lexicon]

# Filted lexicon 371 POS, 362 NEG, 733 TOTAL, all words exist in corpus vocab
filted_lexicon = lexicon.loc[in_lexicon]

# Randomly sample 500 words
sample_lexicon = filted_lexicon.sample(500, random_state=44)
sample_lexicon.POLARITY.value_counts() # 249 POS / 251 NEG

# Output sampled lexicon
sample_lexicon.to_csv('data/Lexicon/lexicon_sample500.csv', encoding='utf-8')
pd.read_csv('data/Lexicon/lexicon_sample500.csv', index_col=[0], encoding='utf-8')
