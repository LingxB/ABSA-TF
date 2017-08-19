from collections import Counter

def freq_dist(tokens):
    c = Counter()
    for t in tokens:
        c.update(t)
    return c


def w_index(counter, start_idx=3):
    w_idx = {w: i + start_idx for i, (w, c) in enumerate(counter.most_common())}
    return w_idx

def df2feats(df, colname, w_idx):
    data = df[colname].apply(lambda x: [w_idx[w] for w in x]).values
    return data