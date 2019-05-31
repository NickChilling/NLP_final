import pickle
import numpy as np
import os
from tqdm import trange
import tensorflow as tf 

from data import process_raw 
from data import make_dataset
from model import BilstmCrfModel

def evaluate(config):

    if not os.path.exists(config.test1_data_path):
        try:
            process_raw(config)
        except Exception as e:
            print(e)
            os.remove(config.test1_data_path)
            exit()
        

    print('Preparing evaluate dataset....')

    dataset_eva,l,total = make_dataset(config.test1_data_path,config)
    dataset_eva = dataset_eva.batch(total)
    eva_iter = dataset_eva.make_one_shot_iterator().get_next()
    print('size of test set: {}'.format(total))

    # loading model
    model = BilstmCrfModel(config)
    model.build()

    global_step = tf.Variable(0, trainable=False, name='global_step')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Session has been initiated')
        ckpt = tf.train.get_checkpoint_state(config.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('Find checkpoint failed')
        print('Checkpoint has been loaded')
        print('Step: {}'.format(global_step.eval()))
        data = sess.run(eva_iter)
        print('Shape of data:', data.shape)
        l = (data.shape[1] - 1) // 2
        sequences = data[:, :l]
        labels = data[:, l:-1]
        sequence_lengths = data[:, -1]
        feed_dict = {model.inputs:sequences,model.lengths: sequence_lengths, model.dr:1}
        # predict = sess.run(model.logits,feed_dict=feed_dict)
        logits,trans_params = sess.run([model.logits, model.trans_params], feed_dict=feed_dict)
        #print(np.array(logits[:sequence_lengths[0]]).shape)
        #print(np.array(trans_params).shape)
        querys = []
        responses = []
        for logit, seq_len, sequence in zip(logits,sequence_lengths, sequences):
            logit_actu = logit[:seq_len]
            vitb_seq,_ = tf.contrib.crf.viterbi_decode(logit_actu,trans_params)
            querys.append(list(sequence[:seq_len]))
            responses.append(vitb_seq)

    if config.id2char_path is None or not os.path.exists(config.id2char_path):
        raise FileNotFoundError('id2char is needed')
    else:
        id2char = pickle.load(open(config.id2char_path, 'rb'))
        id2char.append('?')

    if config.id2tag_path is None or not os.path.exists(config.id2tag_path):
        raise FileNotFoundError('id2tag is needed')
    else:
        id2tag = pickle.load(open(config.id2tag_path, 'rb'))
        id2tag.append('?')
    #print(len(querys), len(querys[0]), len(responses), len(responses[0]))
    #print(querys[0])
    #print(responses[0])
    assert(len(querys) == len(responses))
    for i in range(len(querys)):
        for j in range(len(querys[i])):
            querys[i][j] = id2char[int(querys[i][j])]

    for i in range(len(responses)):
        for j in range(len(responses[i])):
            responses[i][j] = id2tag[int(responses[i][j])]

    print(querys[0])
    print(responses[0])
    print(querys[1])
    print(responses[1])
    if config.task == 'wordseg':
        out_file = open('wordseg_test.txt', 'w')
        for i in range(len(responses)):
            first = True
            word = []
            for q, r in zip(querys[i], responses[i]):
                if first:
                    r = 'B'
                if r == 'B':
                    out_file.write(''.join(word))
                    if not first:
                        out_file.write(' ')
                    word = [q]
                else:
                    word.append(q)
                first = False
            out_file.write('\n')
    elif config.task == 'pos':
        out_file = open('pos_test.txt', 'w')
        for i in range(len(responses)):
            first = True
            correct = True
            word = []
            word_start = '?'
            for q, r in zip(querys[i], responses[i]):
                if first and r[0] == 'I':
                    r = 'B' + r[1:]
                if r == 'O':
                    if word_start == 'O':
                        word.append(q)
                    elif word_start == '?':
                        word_start = r
                        word.append(q)
                    else:
                        out_file.write(''.join(word) + (('/'+word_start[2:]) if word_start != 'O' else '') + ' ')
                        word = [q]
                        word_start = 'O'
                elif r[0] == 'B':
                    out_file.write(''.join(word))
                    if not first:
                        out_file.write((('/'+word_start[2:]) if word_start != 'O' else '') + ' ')
                    word = [q]
                    word_start = r
                else:
                    word.append(q)
                    if r[1:] != word_start[1:]:
                        correct = False
                first = False
            out_file.write('\n')
            if not correct:
                print('Wrong sentence:', querys[i])
                print('    its tag:', responses[i])
    else:
        raise Exception('task should be wordseg or pos')
