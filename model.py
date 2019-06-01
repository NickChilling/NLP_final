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
            self.embedding = tf.Variable(tf.random_normal([self.config.voc + 1, self.config.embedding_dim], dtype=tf.float32), name='embedding')
            #embedded input with shape [batch_size, max_length, embedding_dim]
            embedded = tf.nn.embedding_lookup(self.embedding, self.inputs, name='embedded')  
            embedded = tf.nn.dropout(embedded, self.dr)
        
        with tf.name_scope('lstm-layer'):
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_units)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_units)
            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, embedded, sequence_length=self.lengths, dtype=tf.float32)
            #lstm output with shape [batch_size, max_length, config.lstm_units * 2]
            self.bilstm_output_dr = tf.concat([fw_output, bw_output], -1)
            #self.bilstm_output_dr = tf.nn.dropout(self.bilstm_output, self.dr)
        
        with tf.name_scope('projection'):
            n_steps = tf.shape(self.bilstm_output_dr)[1]
            output = tf.reshape(self.bilstm_output_dr, [-1, self.config.lstm_units * 2])
            self.w = tf.Variable(tf.random_normal([self.config.lstm_units * 2, self.config.n_tags], dtype=tf.float32), name='w')
            self.b = tf.Variable(tf.zeros([self.config.n_tags], dtype=tf.float32), name='b')
            self.logits = tf.reshape(tf.nn.xw_plus_b(output, self.w, self.b), [-1, n_steps, self.config.n_tags],)

        with tf.name_scope('crf-loss'):
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.tags, self.lengths)
            self.loss = -tf.reduce_mean(log_likelihood)

