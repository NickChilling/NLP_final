import tensorflow as tf 


class BilstmCrfModel(object):
    '''BiLSTM-CRF model for sequence tagging'''
    def __init__(self, config):
        self.config = config
    
    def build(self):
        #sequences with shape [batch_size, max_length]
        self.inputs = tf.placeholder(tf.int32, [None, None], name='inputs')  
        #tags with shape [batch_size, max_length]
        self.tags = tf.placeholder(tf.int32, [None, None], name='outputs') 
         #lengths with shape [batch_size]
        self.lengths = tf.placeholder(tf.int32, [None], name='lengths') 
        self.dr = tf.placeholder(tf.float32, [], name='droupout_rate') 
        
        with tf.name_scope('embedding-layer'):
            embedding = tf.random_uniform([self.config.voc + 1, self.config.embedding_dim], name='embedding')
            #embedded input with shape [batch_size, max_length, embedding_dim]
            embedded = tf.nn.embedding_lookup(embedding, self.inputs, name='embedded')  
        
        with tf.name_scope('lstm-layer'):
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_units)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_units)
            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded, sequence_length=self.lengths, dtype=tf.float32)
            #lstm output with shape [batch_size, max_length, config.lstm_units * 2]
            output = tf.concat([fw_output, bw_output], -1)
            output = tf.nn.dropout(output, self.dr)
        
        with tf.name_scope('projection'):
            n_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, self.config.lstm_units * 2])
            w = tf.random_uniform([self.config.lstm_units * 2, self.config.n_tags], dtype=tf.float32)
            b = tf.zeros([self.config.n_tags])#能不能改啊
            self.logits = tf.reshape(tf.nn.xw_plus_b(output, w, b), [-1, n_steps, self.config.n_tags],)

        with tf.name_scope('crf-loss'):
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.tags, self.lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

