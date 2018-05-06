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
MAX_Q = 30
MIN_A = 1
MAX_A = 30
UNK = 'UNK'
START = '$'
END = '%'
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
                    if msgs[i].find('2015') != -1 or msgs[i].find('2016') != -1 or msgs[i].find('2017') != -1 or msgs[i].find('2018') != -1:
                        name = msgs[i].find(class_='user').contents[0]
                        message = msgs[i].next_sibling.contents[0] if msgs[i].next_sibling.contents else None
                        if message and message.find('<p>') == -1:
                            message = filter_whitelist(message.lower())
                            #add to data only when switch
                            if name != AI_NAME:
                                if currQ and currA: #Q and A filled, add to data
                                    currQ = START + ' ' + currQ + ' ' + END
                                    currA = START + ' ' + currA + ' ' + END
                                    if filter_length(currQ, currA):
                                        data_q.append(currQ)
                                        data_a.append(currA)
                                        # print(currQName + ": " + currQ)
                                        # print(AI_NAME + ": " + currA)
                                        # print(" ")
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
    print("Number of conversations used: " + str(len(data_q)))
    return data_q, data_a

def index_words(tokens):
    freq = nltk.FreqDist(itertools.chain(*tokens))
    vocab = freq.most_common(VOCAB_SIZE-3)
    print(vocab[:100])
    i2w = [START] + [END] + [UNK] + [word[0] for word in vocab]
    w2i = dict([(w,i) for i,w in enumerate(i2w)])
    return i2w, w2i, freq

def zero_pad(q_tokens, a_tokens, w2i):
    length = len(q_tokens)

    e_in = np.zeros([length, MAX_Q, VOCAB_SIZE], dtype=np.int32)
    d_in = np.zeros([length, MAX_A, VOCAB_SIZE], dtype=np.int32)

    for i in range(length):
        curr_q = q_tokens[i]
        curr_a = a_tokens[i]

        for j in range(len(curr_q)):
            #check if word is in vocab
            q_vocab_index = w2i[curr_q[j]] if curr_q[j] in w2i else w2i[UNK]
            e_in[i, j, q_vocab_index] = 1

            a_vocab_index = w2i[curr_a[j]] if curr_a[j] in w2i else w2i[UNK]
            d_in[i, j, a_vocab_index] = 1

    return e_in, d_in

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
    e_in, d_in = zero_pad(q_tokens, a_tokens, w2i)
    print("Saving index values")

    np.save('encoder_input.npy', e_in)
    np.save('decoder_input.npy', d_in)
