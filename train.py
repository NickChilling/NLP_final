import numpy as np
import os
import tensorflow as tf 
from tqdm import trange 

from data_utils import process_raw
from data_utils import make_dataset
from models.bilstm_crf import BilstmCrfModel


def train(config):
    #preprocessing raw data if without training data
    if True:
    #if not os.path.exists(config.train_data_path):
        try:
            process_raw(config)
        except Exception as e:
            os.remove(config.train_data_path)
            os.remove(config.dev_data_path)
            os.remove(config.test_data_path)
            raise(e)
    
    print('Preparing dataset...')
    #generate train and dev dataset
    dataset_train, l, _ = make_dataset(config.train_data_path, config)
    dataset_train = dataset_train.shuffle(buffer_size=1000)
    dataset_train = dataset_train.batch(config.batch_size)
    dataset_train = dataset_train.repeat()
    train_iter = dataset_train.make_one_shot_iterator().get_next()
    dataset_dev, _, _ = make_dataset(config.dev_data_path, config, l)
    dataset_dev = dataset_dev.batch(1000)
    dataset_dev = dataset_dev.repeat()
    dev_iter = dataset_dev.make_one_shot_iterator().get_next()
    print('Done')

    print('Initiate model...')
    model = BilstmCrfModel(config)
    model.build()
    print('Done')
    
    with tf.name_scope('optimization'):
        optimizer = tf.train.AdamOptimizer(model.config.lr)
        '''grads_and_vars = optimizer.compute_gradients(model.loss, tf.trainable_variables())
        tf.clip_by_value(grads_and_vars, -5, 5)
        model.train_op = optimizer.apply_gradients(grads_and_vars)'''
        model.train_op = optimizer.minimize(model.loss)
    
    with tf.name_scope('summary'):
        tf.summary.scalar('train_loss', model.loss)
        merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver()

    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_increment = tf.assign_add(global_step, 1)

    print('Initiate a session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(config.tensorboard_path, sess.graph)

        ckpt = tf.train.get_checkpoint_state(config.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        print('Done')

        print('Start training...')
        best_performance = 1e08
        non_increasing_epoch = 0
        for i in trange(config.total_step - sess.run(global_step)):
            data = sess.run(train_iter)
            data = np.array(data, dtype=np.int32)
            data = np.squeeze(data)
            l = (data.shape[1] - 1) // 2
            sequences = data[:, :l]
            labels = data[:, l:-1]
            sequence_lengths = data[:, -1]
            
            feed_dict = {model.inputs:sequences, model.tags:labels, model.lengths:sequence_lengths, \
                model.lengths:sequence_lengths, model.dr:config.dr}
            
            _, step = sess.run([model.train_op, global_step], feed_dict = feed_dict)
            sess.run(global_step_increment)
            
            if step % config.check_freq == 0:
                summary = sess.run(merged_summary, feed_dict = feed_dict)
                writer.add_summary(summary, step)
                data = sess.run(dev_iter)
                sequences = data[:, :l]
                labels = data[:,l:-1]
                sequence_lengths = data[:, -1]
                feed_dict = {model.inputs:sequences, model.tags:labels, model.lengths:sequence_lengths, \
                    model.lengths:sequence_lengths, model.dr:1}
                loss_dev = sess.run(model.loss, feed_dict = feed_dict)
                if loss_dev < best_performance:
                    best_performance = loss_dev
                    non_increasing_epoch = 0
                    saver.save(sess, config.save_path)
                else:
                    non_increasing_epoch += 1
                    if non_increasing_epoch > config.early_stopping:
                        print('early_stopping')
                        break

        





    
    
    