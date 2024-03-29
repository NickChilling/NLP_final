import tensorflow as tf 
import os

from train import train
from evaluate import evaluate
#NOTE run word_seg.py
flags = tf.app.flags
#NOTE setting flags 

flags.DEFINE_integer('embedding_dim', 100, 'dimension of the character embedding')
flags.DEFINE_integer('lstm_units', 50, 'num_units of the BiLSTM layer')
flags.DEFINE_integer('n_tags', 4, 'num of tags')
flags.DEFINE_integer('batch_size', 8, 'batch size for training')
flags.DEFINE_integer('early_stopping', 3, 'non increasing epochs for early stopping')
flags.DEFINE_integer('total_step', 10000, 'total step of training')
flags.DEFINE_integer('check_freq', 100, 'total step of training')

flags.DEFINE_integer('voc', 2095, 'volume of the vocabulary')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('dr', 0.65, 'dropout keep rate')
flags.DEFINE_string('task', 'wordseg', 'task: "wordseg" or "pos"')

flags.DEFINE_string('mode', 'train', 'running mode: "train", "evaluate"')

flags.DEFINE_string('model', 'bilstm-crf', 'bilstm-crf or self-attention-crf')
flags.DEFINE_string('raw_train_data_path', r'./data/trainset/train_cws.txt', 'path of the raw train data')
flags.DEFINE_string('raw_dev_data_path', r'./data/devset/val_cws.txt', 'path of the raw dev data')
flags.DEFINE_string('raw_test1_data_path', r'./data/testset1/test_cws1.txt', 'path of the raw test1 data')
flags.DEFINE_string('train_data_path', r'./data/wordseg/train_data', 'path of the processed train data')
flags.DEFINE_string('dev_data_path', r'./data/wordseg/dev_data', 'path of the processed dev data')
flags.DEFINE_string('test1_data_path', r'./data/wordseg/test1_data', 'path of the processed test1 data')
flags.DEFINE_string('char2id_path', r'./data/wordseg/char2id.pkl', 'path of the char2id, serialized by pickle')
flags.DEFINE_string('tag2id_path', r'./data/wordseg/tag2id.pkl', 'path of the tag2id, serialized by pickle')
flags.DEFINE_string('id2char_path', r'./data/wordseg/id2char.pkl', 'path of the id2char, serialized by pickle')
flags.DEFINE_string('id2tag_path', r'./data/wordseg/id2tag.pkl', 'path of the id2tag, serialized by pickle')

flags.DEFINE_string('max_length_path', r'./data/wordseg/max_length.pkl', 'path of the max_length, serialized by pickle')
flags.DEFINE_string('save_path', r'./ckpt/wordseg', 'path to save checkpoints')
flags.DEFINE_string('tensorboard_path', r'./tensorboard/wordseg', 'path for tensorboard')


config = flags.FLAGS

def main(_):
    config.save_path += 'batch_size{},learning_rate{},lstm_units{},embedding_dim{}/'.format(config.batch_size, config.lr, config.lstm_units, config.embedding_dim)
    config.tensorboard_path += 'batch_size{},learning_rate{},lstm_units{},embedding_dim{}/'.format(config.batch_size, config.lr, config.lstm_units, config.embedding_dim)
    print('ckpt save path: ', config.save_path)
    print('tensorboard logdir: ', config.tensorboard_path)
    if not os.path.exists('data/wordseg'):
        os.mkdir('data/wordseg')# batch_size_{}_learning_rate_{}_lstm_units_{}_embedding_dim_{}_early_stop_200/
    config.save_path += 'batch_size_{}_learning_rate_{}_lstm_units_{}_embedding_dim_{}/'.format(config.batch_size, config.lr, config.lstm_units, config.embedding_dim)
    print('ckpt save path: ', config.save_path)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    #NOTE choose mode according to config.mode
    if config.mode == 'train':
        train(config)
    elif config.mode == 'evaluate':
        evaluate(config)
    elif config.mode == 'test':

        config.raw_test1_data_path = './test2.txt'
        config.test1_data_path = './test2_data'
        evaluate(config)
    else:
        raise ValueError('Invalid mode {}'.format(config.mode))

    
if __name__ == '__main__':
    tf.app.run()



