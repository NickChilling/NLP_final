import tensorflow as tf 

from train import train
from test import test
from demonstrate import demonstrate

flags = tf.app.flags

flags.DEFINE_integer('embedding_dim', 100, 'dimension of the character embedding')
flags.DEFINE_integer('lstm_units', 100, 'num_units of the BiLSTM layer')
flags.DEFINE_integer('n_tags', 13, 'dimension of the character embedding')
flags.DEFINE_integer('batch_size', 50, 'batch size for training')
flags.DEFINE_integer('early_stopping', 10, 'non increasing epochs for early stopping')
flags.DEFINE_integer('total_step', 100000, 'total step of training')
flags.DEFINE_integer('check_freq', 500, 'total step of training')
flags.DEFINE_integer('voc', 2428, 'volume of the vocabulary')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('dr', 0.5, 'dropout rate')
flags.DEFINE_string('mode', 'train', 'train, test or demonstrate')
flags.DEFINE_string('model', 'bilstm-crf', 'bilstm-crf or self-attention-crf')
flags.DEFINE_string('raw_data_path', r'.\data\BosonNLP_NER_6C.txt', 'path of the train data')
flags.DEFINE_string('train_data_path', r'.\data\train_data', 'path of the train data')
flags.DEFINE_string('dev_data_path', r'.\data\dev_data', 'path of the dev data')
flags.DEFINE_string('test_data_path', r'.\data\test_data', 'path of the test data')
flags.DEFINE_string('char2id_path', r'.\data\char2id.pkl', 'path of the test data')
flags.DEFINE_string('tag2id_path', r'.\data\tag2id.pkl', 'path of the test data')
flags.DEFINE_string('id2char_path', r'.\data\id2char.pkl', 'path of the test data')
flags.DEFINE_string('id2tag_path', r'.\data\id2tag.pkl', 'path of the test data')
flags.DEFINE_string('save_path', r'.\model\checkpoints', 'path of saved checkpoints')
flags.DEFINE_string('tensorboard_path', r'.\tensorboard', 'path of tensorboard')

config = flags.FLAGS


'''class Config(object):
    def __init__(self):
        self.embedding_dim = flags.embedding_dim
        self.lr = flags.lr
        self.mode = flags.mode
        self.model = flags.model
        self.raw_data_path = flags.raw_data_path
        self.train_data_path = flags.train_data_path
        self.dev_data_path = flags.dev_data_path
        self.test_data_path = flags.test_data_path'''

def main(_):
    if config.mode == 'train':
        train(config)
    elif config.mode == 'test':
        test(config)
    elif config.mode == 'demonstrate':
        demonstrate(config)
    else:
        raise ValueError('Invalid mode {}'.format(config.mode))

    
if __name__ == '__main__':
    try:
        tf.app.run()
    except SystemExit:
        print('Done')

