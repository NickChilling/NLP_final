import tensorflow as tf 
import os

from train import train
from evaluate import evaluate

flags = tf.app.flags

flags.DEFINE_integer('embedding_dim', 200, 'dimension of the character embedding')
flags.DEFINE_integer('lstm_units', 200, 'num_units of the BiLSTM layer')
flags.DEFINE_integer('n_tags', 66, 'num of tags')
flags.DEFINE_integer('batch_size', 50, 'batch size for training')
flags.DEFINE_integer('early_stopping', 10, 'non increasing epochs for early stopping')
flags.DEFINE_integer('total_step', 100000, 'total step of training')
flags.DEFINE_integer('check_freq', 500, 'total step of training')
flags.DEFINE_integer('voc', 2095, 'volume of the vocabulary')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('dr', 0.5, 'dropout rate')
flags.DEFINE_string('task', 'pos', 'task: "wordseg" or "pos"')
flags.DEFINE_string('mode', 'evaluate', 'running mode: "train", "evaluate" or "all"')
flags.DEFINE_string('model', 'bilstm-crf', 'bilstm-crf or self-attention-crf')
flags.DEFINE_string('raw_train_data_path', r'./data/trainset/train_pos.txt', 'path of the raw train data')
flags.DEFINE_string('raw_dev_data_path', r'./data/devset/val_pos.txt', 'path of the raw dev data')
flags.DEFINE_string('raw_test1_data_path', r'./data/testset1/test_pos1.txt', 'path of the raw test1 data')
flags.DEFINE_string('train_data_path', r'./data/pos/train_data', 'path of the processed train data')
flags.DEFINE_string('dev_data_path', r'./data/pos/dev_data', 'path of the processed dev data')
flags.DEFINE_string('test1_data_path', r'./data/pos/test1_data', 'path of the processed test1 data')
flags.DEFINE_string('char2id_path', r'./data/pos/char2id.pkl', 'path of the char2id, serialized by pickle')
flags.DEFINE_string('tag2id_path', r'./data/pos/tag2id.pkl', 'path of the tag2id, serialized by pickle')
flags.DEFINE_string('id2char_path', r'./data/pos/id2char.pkl', 'path of the id2char, serialized by pickle')
flags.DEFINE_string('id2tag_path', r'./data/pos/id2tag.pkl', 'path of the id2tag, serialized by pickle')
flags.DEFINE_string('save_path', r'./ckpt/pos', 'path to save checkpoints')
flags.DEFINE_string('tensorboard_path', r'./tensorboard/pos', 'path for tensorboard')

config = flags.FLAGS

def main(_):
    if not os.path.exists('data/pos'):
        os.mkdir('data/pos')
    if config.mode == 'train':
        train(config)
    elif config.mode == 'evaluate':
        evaluate(config)
    elif config.mode == 'all':
        train(config)
        evaluate(config)
    else:
        raise ValueError('Invalid mode {}'.format(config.mode))

    
if __name__ == '__main__':
    try:
        tf.app.run()
    #except SystemExit:
    except Exception as e:
        print(e)
        print('Done')

