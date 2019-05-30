
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

    dataset_eva,l,_ = make_dataset(config.test1_data_path,config)
    dataset_eva = dataset_eva.shuffle(buffer_size=1000)
    dataset_eva = dataset_eva.batch(config.batch_size)
    dataset_eva = dataset_eva.repeat()
    eva_iter = dataset_eva.make_one_shot_iterator().get_next()
    print('giving a iteration')

    # loading model
    result = []
    model = BilstmCrfModel(config)
    model.build()
    with tf.Session() as sess:
        saver =tf.train.import_meta_graph('./ckpt/{0}.meta'.format(config.task))
        saver.restore(sess,config.save_path) #TODO check out
        graph = tf.get_default_graph()
    #     data = sess.run(eva_iter)
    #     sequences = data[:,:1]
    #     sequence_length = data[:,-1]
    #     feed_dict = {model.inputs:sequences,model.lengths: sequence_length}
    #     # predict = sess.run(model.logits,feed_dict=feed_dict)
    #     logits,trans_params = sess.run([model.logits, model.trans_params], feed_dict=feed_dict)
    #     for logit,seq_len in zip(logits,sequence_length):
    #         logit_actu = logit[:seq_len]
    #         vitb_seq,_ = tf.contrib.crf.vitebi_decode(logit_actu,trans_params)
    #         result.append(vitb_seq)
    # print(result)
    # return result
        



