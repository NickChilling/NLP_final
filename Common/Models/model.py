
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

    

