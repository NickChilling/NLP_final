import tensorflow as tf

class BaseModel(object):
    '''Interface for sequence tagging models
    '''
    
    def build_graph(self, hyperparams, scope=None):
        '''Creates a sequence tagging model with tensorflow's computation graph
            Args:
                hyperparams: a dict of hyperparameters for the model
                scope: the variable scope prefixed by model's nodes, using a default one if not provided
        '''
        raise NotImplementedError('Interface "build_graph(self, hyperparams, scope = None)" should be implemented')

    def train(self, hyperparams, sample_generator):
        '''execute the train process
            Args:
                hyperparams: a dict of hyperparameters for the training process
                sample_generator: a generator of training samples  
        '''
        raise NotImplementedError('Interface "train(self, hyperparams, sample_generator)" should be implemented')
    
    def test(self, sample_generator=None):
        '''execute the test process
            Args:
                sample_generator: a generator of training samples, set None if third party tool is used.  
        '''
        pass

class WordSegModel(BaseModel):
    def __init__(self, config):
        self.config = config

    def build_graph(self, hyperparams, scope=None):
        self.inputs = tf.placeholder(tf.int32,[None,None],name = 'inputs')
        self.tags = tf.placeholder(tf.int32,[None,None],name='outputs')
        self.lengths = tf.placeholder(tf.int32,[None],name = 'length')
        self.drop_rate = tf.placeholder(tf.float32,[],name='dropout_rate')

        with tf.name_scope('embedding_layer'):
            embedding = tf.random_uniform([self.config.voc+1,self.config.embedding_dim],name='embedding')
            embedded = tf.nn.embedding_lookup(embedding,self.inputs,name = 'embedded')

        with tf.name_scope('lstm_layer'):
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_units)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_units)
            (fw_output,bw_output),_ = tf.nn.bidirectional_dynamic_rnn() #TODO  deprecated function
            output = tf.concat([fw_output,bw_output],-1)
            output = tf.nn.dropout(output,self.drop_rate)
        
        with tf.name_scope('projection'):
            n_steps = tf.shape(output)[1]
            output = tf.reshape(output,[-1,self.config.lstm_units*2])
            
    def train(self, hyperparams, sample_generator):
        return super().train(hyperparams, sample_generator)
    def test(self, sample_generator=None):
        return super().test(sample_generator=sample_generator)

    

