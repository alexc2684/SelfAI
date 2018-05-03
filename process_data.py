from bs4 import BeautifulSoup as bs
import os
import nltk
import numpy as np
import itertools

AI_NAME = 'Alex Chan'
WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
MSG_PATH = 'messages/'
MIN_Q = 0
MAX_Q = 20
MIN_A = 1
MAX_A = 20
UNK = 'UNK'
EOS = 'EOS'
GO = 'GO'
PAD = '_'#'PAD'
VOCAB_SIZE = 6000


def filter_whitelist(sentence):
    return ''.join([ch for ch in sentence if ch in WHITELIST])

def filter_length(q, a):
    qlen = q.split(' ')
    alen = a.split(' ')
    return len(qlen) > MIN_Q and len(qlen) < MAX_Q and len(alen) > MIN_A and len(alen) < MAX_A

def process_data(MSG_PATH):
    data_q = []
    data_a = []

    count = 0
    for msg in os.listdir(MSG_PATH):
        currQ = ''
        currQName = ''
        currA = ''
        if msg.endswith('html'):
            path = MSG_PATH + msg
            with open(path, 'r') as f:
                html = bs(f, 'html.parser')
                msgs = html.find_all(class_='message')
                # print(msgs[0].next_sibling)
                for i in range(len(msgs)-1,-1,-1):
                    name = msgs[i].find(class_='user').contents[0]
                    message = msgs[i].next_sibling.contents[0] if msgs[i].next_sibling.contents else None
                    if message and message.find('<p>') == -1:
                        message = filter_whitelist(message.lower())
                        #add to data only when switch
                        if name != AI_NAME:
                            if currQ and currA: #Q and A filled, add to data
                                if filter_length(currQ, currA):
                                    data_q.append(currQ)
                                    data_a.append(currA)
                                    currQ = ''
                                    currQName = ''
                                    currA = ''
                            if name == currQName:
                                currQ += ' ' + message
                            else:
                                currQName = name
                                currQ = message
                        else:
                            if currA:
                                currA += ' ' + message
                            else:
                                currA = message
        if count > 2:
            break
        count += 1
    print("Number of conversations used: " + len(data_q))
    return data_q, data_a

def index_words(tokens):
    freq = nltk.FreqDist(itertools.chain(*tokens))
    vocab = freq.most_common(VOCAB_SIZE)
    print(vocab[:100])
    i2w = [PAD] + [UNK] + [word[0] for word in vocab]
    w2i = dict([(w,i) for i,w in enumerate(i2w)])
    return i2w, w2i, freq

def zero_pad(q_tokens, a_tokens, w2i):
    length = len(q_tokens)

    idx_q = np.zeros([length, MAX_Q], dtype=np.int32)
    idx_a = np.zeros([length, MAX_A], dtype=np.int32)

    for i in range(length):
        q = sent_to_indices(q_tokens[i], w2i, MAX_Q)
        a = sent_to_indices(a_tokens[i], w2i, MAX_A)
        idx_q[i] = np.array(q)
        idx_a[i] = np.array(a)

    return idx_q, idx_a

def sent_to_indices(sent, w2i, max_length):
    indices = []
    for word in sent:
        if word in w2i:
            indices.append(w2i[word])
        else:
            indices.append(w2i[UNK])

    return indices + [0 for i in range(max_length - len(indices))]

if __name__ == "__main__":
    print("Filtering data")
    q_data, a_data = process_data(MSG_PATH)
    print("Finished filtering data")
    q_tokens = [sent.split(' ') for sent in q_data]
    a_tokens = [sent.split(' ') for sent in a_data]
    print("Creating index word mappings")
    i2w, w2i, freqs = index_words(q_tokens + a_tokens)
    idx_q, idx_a = zero_pad(q_tokens, a_tokens, w2i)
    print("Saving index values")

    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
