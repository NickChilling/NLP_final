import numpy as np
import os
import tensorflow as tf 
from tqdm import trange 

from data import process_raw
from data import make_dataset
from model import BilstmCrfModel


def train(config):
    #preprocessing raw data if not done
    #if True:
    if not os.path.exists(config.train_data_path):
        try:
            process_raw(config)
        except Exception as e:
            print(e)
            os.remove(config.train_data_path)
            os.remove(config.dev_data_path)
            os.remove(config.test_data_path)
            exit()
    
    print('Preparing dataset...')
    #generate train and dev dataset
    dataset_train, l, train_total = make_dataset(config.train_data_path, config)
    dataset_train = dataset_train.shuffle(buffer_size=1000)
    dataset_train = dataset_train.batch(config.batch_size)
    dataset_train = dataset_train.repeat()
    train_iter = dataset_train.make_one_shot_iterator().get_next()
    dataset_dev, _, dev_total = make_dataset(config.dev_data_path, config, l)
    dataset_dev = dataset_dev.batch(dev_total)
    dataset_dev = dataset_dev.repeat()
    dev_iter = dataset_dev.make_one_shot_iterator().get_next()
    print('Size of training set', train_total)
    print('Size of dev set', dev_total)
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

    #saver = tf.train.Saver(max_to_keep=3)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    global_step_increment = tf.assign_add(global_step, 1)
    step_holder = tf.placeholder(tf.int32, [], name='step_holder')
    global_step_assign = tf.assign(global_step, step_holder)

    saver = tf.train.Saver(max_to_keep=2)

    print('Initiate a session')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(config.tensorboard_path, sess.graph)

        ckpt = tf.train.get_checkpoint_state(config.save_path)
        print('ckpt: ', ckpt, ckpt.model_checkpoint_path if ckpt else None)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #sess.run(global_step_assign, feed_dict={step_holder:int(ckpt.model_checkpoint_path[ckpt.model_checkpoint_path.rfind('-')+1:])})
            print('Checkpoint has been loaded')
            print('From step {}, start?(y/n):'.format(global_step.eval()))
            yorn = input().strip()
            if yorn != 'y':
                exit()
        else:
            print('No checkpoint, start from scratch?(y/n):')
            yorn = input().strip()
            if yorn != 'y':
                exit()
        print('Done')

        print('Start training...')
        best_performance = 1e08
        non_increasing_epoch = 0
        init_step = sess.run(global_step)
        for i in trange(init_step, config.total_step):
            #print(i, sess.run(global_step))
            data = sess.run(train_iter)
            data = np.array(data, dtype=np.int32)
            data = np.squeeze(data)
            l = (data.shape[1] - 1) // 2
            sequences = data[:, :l]
            labels = data[:, l:-1]
            sequence_lengths = data[:, -1]
            
            feed_dict = {model.inputs:sequences, model.tags:labels, model.lengths:sequence_lengths, \
                model.dr:config.dr}
            
            _, step = sess.run([model.train_op, global_step], feed_dict = feed_dict)
            sess.run(global_step_increment)
            
            if step % config.check_freq == 0:
                summary = sess.run(merged_summary, feed_dict = feed_dict)
                writer.add_summary(summary, step)
                data = sess.run(dev_iter)
                print('Shape of dev data:', data.shape)
                sequences = data[:, :l]
                labels = data[:,l:-1]
                sequence_lengths = data[:, -1]
                feed_dict = {model.inputs:sequences, model.tags:labels, model.lengths:sequence_lengths, \
                    model.dr:1}
                loss_dev = sess.run(model.loss, feed_dict = feed_dict)
                print('dev_loss({}):'.format(step), loss_dev)
                dev_file = open(config.save_path+'dev_loss', 'a')
                print('dev_loss({}):'.format(step), loss_dev, file=dev_file)
                dev_file.close()
                if loss_dev < best_performance-1:
                    best_performance = loss_dev
                    non_increasing_epoch = 0
                    if i > init_step:
                        saver.save(sess, config.save_path, global_step=step)
                        print('model has been saved in {}'.format(config.save_path))
                        print('model has been saved in {}'.format(config.save_path))
                        print('model has been saved in {}'.format(config.save_path))
                else:
                    non_increasing_epoch += 1
                    if non_increasing_epoch > config.early_stopping:
                        print('early_stopping')
                        break

        





    
    
    
