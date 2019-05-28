from collections import defaultdict
from collections import Iterable
import numpy as np 
import pickle
import string
import tensorflow as tf 


tag2label = {'time':'TIM', 'location':'LOC', 'person_name':'PER', 'org_name':'ORG', 'company_name':'COM', 'product_name':'PRO'}


def process_raw(config):
    '''transform raw data into BIO form'''

    print('Start processing raw data...')
    with open(config.raw_train_data_path, 'r', encoding='utf-8') as raw_train, \
        open(config.raw_dev_data_path, 'r', encoding='utf-8') as raw_dev, \
        open(config.raw_test1_data_path, 'r', encoding='utf-8') as raw_test, \
        open(config.train_data_path, 'w', encoding='utf-8') as train, \
        open(config.dev_data_path, 'w', encoding='utf-8') as dev, \
        open(config.test1_data_path, 'w', encoding='utf-8') as test:
        for f, g in ((raw_train, train), (raw_dev, dev), (raw_test, test)):
            if config.task == 'pos':
                for line in f:
                    '''
                    if count == batch_size:
                        batches.append(samples)
                        samples = []
                        count = 0'''
                    charactors = []
                    tags = []
                    orig = line
                    line = line.strip().split()
                    for unit in line:
                        '''if unit == '$$_' or unit == '$$__':
                            charactors.extend(list(unit))
                            tags.extend('O' * len(unit))
                            continue'''
                        slash_idx = unit.rfind('/')
                        word = unit[:slash_idx]
                        pos = unit[slash_idx+1:]
                        if slash_idx == -1 or len(pos) > 3 or len(pos) == 0 or pos[0].isupper() or unit == '$$_' or unit == '$$__':
                            charactors.extend(list(unit))
                            tags.extend('O' * len(unit))
                            #print(orig)
                            #print(unit)
                            continue
                        for i, c in enumerate(word):
                            charactors.append(c)
                            if i == 0:
                                tags.append('B-{}'.format(pos.upper()))
                            else:
                                tags.append('I-{}'.format(pos.upper()))
                    if len(charactors) != len(tags):
                        print(orig)
                        print(charactors)
                        print(tags)
                        raise Exception('len(charactors) != len(tags)')
                    for charactor, tag in zip(charactors, tags):
                        print(charactor, tag, file=g)
                    print(file=g)
            elif config.task == 'wordseg':
                raise NotImplementedError()
            else:
                raise Exception('Wrong argument: task should be "wordseg" or "pos"')
    print('Done')
            
def make_dataset(path, config, max_length=None):
    '''load data from path and return dataset, max_length and voc'''

    X = []
    Y = []
    lengths = []
    sentence = []
    labels = []

    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line != '':
            char, label = line[0], line[2:]
            sentence.append(char)
            labels.append(label)
        else:
            l = len(sentence) if max_length is None else min(max_length, len(sentence))
            if l > 0:
                X.append(sentence)
                Y.append(labels)
                lengths.append(l)
                sentence = []
                labels = []
            else:
                raise Exception
    
    l = max(len(x) for x in X)
    if max_length == None:
        max_length = l
    else:
        max_length = max(max_length, l)
    

    if config.char2id_path is None:
        char2id, id2char = make_char2id(X)
        pickle.dump(char2id, open(r'.\data\char2id.pkl', 'wb'))
        pickle.dump(id2char, open(r'.\data\id2char.pkl', 'wb'))
    else:
        char2id = pickle.load(open(config.char2id_path, 'rb'))
    
    if config.tag2id_path is None:
        tag2id, id2tag = make_tag2id()
        pickle.dump(tag2id, open(r'.\data\tag2id.pkl', 'wb'))
        pickle.dump(id2tag, open(r'.\data\id2tag.pkl', 'wb'))
    else:
        tag2id = pickle.load(open(config.tag2id_path, 'rb'))

    pad_sequences(X, max_length, '空')
    pad_sequences(Y, max_length, 'O')
    
    for i, y in enumerate(Y):
        for j, c in enumerate(y):
            if c not in tag2id:
                raise Exception

    transform_w2id(X, char2id)
    transform_w2id(Y, tag2id)

    X, Y, lengths = (np.array(w) for w in (X, Y, lengths))
    lengths = np.expand_dims(lengths, -1)
    data = np.concatenate((X, Y, lengths), axis=1)
    
    return tf.data.Dataset.from_tensor_slices(data), max_length, len(char2id)

def make_char2id(X):
    '''X is a 2d list
       return a defaultdict char2id and a list id2char'''
    counter = defaultdict(int)
    for s in X:
        for c in s:
            counter[c] += 1
    char2id = {}
    id2char = []
    i = 0
    for c in counter:
        if counter[c] > 5:
            char2id[c] = i
            id2char.append(c)
            i += 1
    return char2id, id2char

def make_tag2id():
    '''return a defaultdict tag2id and a list id2tag'''
    tag2id = {'O':0}
    id2tag = ['O']
    i = 1
    for t in tag2label:
        tag2id['B-'+tag2label[t]] = i
        tag2id['I-'+tag2label[t]] = i + 1
        id2tag.append('B-'+tag2label[t])
        id2tag.append('I-'+tag2label[t])
        i += 2
    return tag2id, id2tag

def transform_w2id(X, w2id, X_is_2d=True):
    '''X is a list of list
       transform elements in X to id with reference to w2id'''
    unknown = len(w2id)
    if X_is_2d:
        for s in X:
            for i, x in enumerate(s):
                s[i] = w2id[x] if x in w2id else unknown
    else:
        for i, x in enumerate(X):
            if isinstance(x, Iterable) and not isinstance(x, str):
                transform_w2id(x, w2id)
            else:
                X[i] = w2id[x] if x in w2id else unknown

def pad_sequences(seqs, length, token):
    '''pad each seq in seqs to length with token'''
    for i, seq in enumerate(seqs):
        if len(seq) < length:
            seq.extend([token] * (length - len(seq)))
        else:
            seqs[i] = seq[:length]
                               
