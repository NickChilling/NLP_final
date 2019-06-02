import pickle
import numpy as np
import os
from tqdm import trange
import tensorflow as tf 
import time

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
    #global_step = tf.Variable(5300, trainable=False, name='global_step')

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
        print('model.embedding')
        print(model.embedding.eval())
        print('model.w')
        print(model.w.eval())
        print('model.b')
        print(model.b.eval())
        l = (data.shape[1] - 1) // 2
        sequences = data[:, :l]
        labels = data[:, l:-1]
        sequence_lengths = data[:, -1]
        feed_dict = {model.inputs:sequences,model.lengths: sequence_lengths, model.dr:1}
        # predict = sess.run(model.logits,feed_dict=feed_dict)
        logits,trans_params = sess.run([model.logits, model.trans_params], feed_dict=feed_dict)
        '''print('trans_params')
        print(trans_params)
        print('model.bilstm_output', model.bilstm_output.eval(feed_dict=feed_dict).shape)
        print(model.bilstm_output.eval(feed_dict=feed_dict))
        print('bilstm_output_dr', model.bilstm_output_dr.eval(feed_dict=feed_dict).shape)
        print(model.bilstm_output_dr.eval(feed_dict=feed_dict))
        print('model.logits')
        print(model.logits.eval(feed_dict=feed_dict))'''
        print('trans_params')
        print(trans_params)
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
        out_file = open('wordseg_test.txt', 'w', encoding='utf-8')
        for i in range(len(responses)):
            first = True
            word = []
            for q, r in zip(querys[i], responses[i]):
                if first and (r == 'I' or r == 'E'):
                    r = 'B'
                if r == 'B':
                    out_file.write(''.join(word))
                    if len(word) > 0:
                        out_file.write(' ')
                    word = [q]
                elif r == 'S':
                    out_file.write(''.join(word))
                    if len(word) > 0:
                        out_file.write(' ')
                    out_file.write(q)
                    out_file.write(' ')
                    word = []
                else:
                    word.append(q)
                first = False
            out_file.write(''.join(word))
            out_file.write('\n')
        print(config.raw_test1_data_path)
        out_file.close()
        word_seg_score(config.raw_test1_data_path,'wordseg_test.txt')
        print('----------------------------------------')
        print('Start wordseg_evaluate.py...')
        time.sleep(1)
        os.system('python wordseg_evaluate.py')
        time.sleep(3)
    elif config.task == 'pos':
        out_file = open('pos_test.txt', 'w', encoding='utf-8')
        out_file_ws = open('wordseg_test.txt', 'w', encoding='utf-8')
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
                        out_file_ws.write(''.join(word) + ' ')
                        word = [q]
                        word_start = 'O'
                elif r[0] == 'B':
                    out_file.write(''.join(word))
                    out_file_ws.write(''.join(word))
                    if len(word) > 0:
                        out_file.write((('/'+word_start[2:]).lower() if word_start != 'O' else '') + ' ')
                        out_file_ws.write(' ')
                    word = [q]
                    word_start = r
                else:
                    word.append(q)
                    if r[1:] != word_start[1:]:
                        correct = False
                first = False
            out_file.write(''.join(word) + (('/'+word_start[2:]).lower() if word_start != 'O' else ''))
            out_file.write('\n')
            out_file_ws.write(''.join(word))
            out_file_ws.write('\n')
            if False:
            #if not correct:
                print('Wrong sentence:', querys[i])
                print('    its tag:', responses[i])
        out_file.close()
        out_file_ws.close()
        print('Start pos_evaluate.py...')
        time.sleep(1)
        os.system('python pos_evaluate.py')
        time.sleep(3)
        print('----------------------------------------')
        print('Start wordseg_evaluate.py...')
        time.sleep(1)
        os.system('python wordseg_evaluate.py')
        time.sleep(3)
    else:
        raise Exception('task should be wordseg or pos')

def word_seg_score(test_path,result_path):
    with open(test_path,'r',encoding='utf-8') as test,open(result_path,'r',encoding='utf-8') as results:
        result_lines = results.readlines()
        test_lines = test.readlines()
        print(len(result_lines),len(test_lines))
        assert len(result_lines)==len(test_lines)
        total_word = 0
        true_word = 0
        predict_word = 0
        for line_index in range(len(result_lines)):
            result_line = result_lines[line_index]
            test_line = test_lines[line_index]
            tc_start = 0
            rc_start = 0
            result_set = set()
            test_set = set()
            while tc_start < len(test_line):
                tc_end = tc_start
                while tc_end<len(test_line) and test_line[tc_end]!=' ':
                    tc_end+=1
                total_word +=1
                test_word = (tc_start,tc_end)
                test_set.add(test_word)
                tc_start = tc_end+1

            while rc_start <len(result_line):
                rc_end = rc_start
                while rc_end<len(result_line) and result_line[rc_end]!=' ':
                    rc_end+=1
                result_word = (rc_start,rc_end)
                result_set.add(result_word)
                rc_start = rc_end+1
            line_true_word = len(test_set&result_set)
            true_word += line_true_word
            line_pred_word = len(result_set)
            predict_word += line_pred_word
        precision = true_word/total_word
        recall = true_word/predict_word
        f1 = 2*precision*recall/(precision+recall)
        print('precision:',precision)
        print('recall:',recall)
        print('f1:',f1)