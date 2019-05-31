from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict
from collections import Iterable
import numpy as np 
import pickle
import string
import tensorflow as tf 
import os



def process_raw(config):
    '''transform raw data into BIO form'''

    print('Start processing raw data...')
    if config.mode == 'test':
        with open(config.raw_test2_data_path) as f, open(config.test2_data_path, 'w') as g:
            for line in f:
                if line.strip() == '':
                    continue
                charactors = []
                tags = []
                for i, c in enumerate(line):
                    charactors.append(c)
                    tags.append('?')
                if len(charactors) != len(tags):
                    print(line)
                    print(charactors)
                    print(tags)
                    raise Exception('len(charactors) != len(tags)')
                for charactor, tag in zip(charactors, tags):
                    print(charactor, tag, file=g)
                print(file=g)
        return
    print('====================================================')
    with open(config.raw_train_data_path, 'r', encoding='utf-8') as raw_train, \
        open(config.raw_dev_data_path, 'r', encoding='utf-8') as raw_dev, \
        open(config.raw_test1_data_path, 'r', encoding='utf-8') as raw_test, \
        open(config.train_data_path, 'w', encoding='utf-8') as train, \
        open(config.dev_data_path, 'w', encoding='utf-8') as dev, \
        open(config.test1_data_path, 'w', encoding='utf-8') as test:
        for f, g in ((raw_train, train), (raw_dev, dev), (raw_test, test)):
            if config.task == 'pos':
                for line in f:
                    if line.strip() == '':
                        continue
                    charactors = []
                    tags = []
                    orig = line
                    line = line.strip().split()
                    for unit in line:
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
                for line in f:
                    if line.strip() == '':
                        continue
                    charactors = []
                    tags = []
                    orig = line
                    line = line.strip().split()
                    for word in line:
                        for i, c in enumerate(word):
                            charactors.append(c)
                            if i == 0:
                                tags.append('B') 
                            else:
                                tags.append('I')
                    if len(charactors) != len(tags):
                        print(orig)
                        print(charactors)
                        print(tags)
                        raise Exception('len(charactors) != len(tags)')
                    for charactor, tag in zip(charactors, tags):
                        print(charactor, tag, file=g)
                    print(file=g)
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
    total = 0

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
                total += 1
            else:
                print(X)
                print(Y)
                raise Exception()
    
    l = max(len(x) for x in X)
    if max_length == None:
        max_length = l
    else:
        max_length = max(max_length, l)
    
    if config.max_length_path is None or not os.path.exists(config.max_length_path):
        pickle.dump(max_length, open(config.max_length_path, 'wb'))
    else:
        l = pickle.load(open(config.max_length_path, 'rb'))
        max_length = max(max_length, l)

    if config.char2id_path is None or not os.path.exists(config.char2id_path):
        char2id, id2char = make_char2id(X)
        pickle.dump(char2id, open(config.char2id_path, 'wb'))
        pickle.dump(id2char, open(config.id2char_path, 'wb'))
    else:
        char2id = pickle.load(open(config.char2id_path, 'rb'))

    if config.tag2id_path is None or not os.path.exists(config.tag2id_path):
        tag2id, id2tag = make_tag2id(Y)
        pickle.dump(tag2id, open(config.tag2id_path, 'wb'))
        pickle.dump(id2tag, open(config.id2tag_path, 'wb'))
    else:
        tag2id = pickle.load(open(config.tag2id_path, 'rb'))

    pad_sequences(X, max_length, '空')
    if config.task == 'pos':
        pad_sequences(Y, max_length, 'O')
    elif config.task == 'wordseg':
        pad_sequences(Y, max_length, 'B') 
    #print('len(tag2id):', len(tag2id))
    #print('len(char2id):', len(char2id))
    for i, y in enumerate(Y):
        for j, c in enumerate(y):
            if c not in tag2id:
                raise Exception()

    transform_w2id(X, char2id)
    transform_w2id(Y, tag2id)

    X, Y, lengths = (np.array(w) for w in (X, Y, lengths))
    lengths = np.expand_dims(lengths, -1)
    data = np.concatenate((X, Y, lengths), axis=1)
    
    return tf.data.Dataset.from_tensor_slices(data), max_length, total

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
        if counter[c] > 3:
            char2id[c] = i
            id2char.append(c)
            i += 1
    return char2id, id2char

def make_tag2id(Y):
    '''Y is a 2d list
       return a defaultdict tag2id and a list id2tag'''
    counter = defaultdict(int)
    for s in Y:
        for c in s:
            counter[c] += 1
    tag2id = {}
    id2tag = []
    i = 0
    for c in counter:
        tag2id[c] = i
        id2tag.append(c)
        i += 1
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
                               
