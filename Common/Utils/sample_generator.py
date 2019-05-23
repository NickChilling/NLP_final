import os
import random

def wordseg_sample_generator(batch_size, prefix):
    '''a sample generator for word segmentation
        Args:
            batch_size: the num of sentence in a batch
            prefix: "train", "dev" or "test"
        
        Yields a list of samples of size "batch_size". 
        Each sample contains two lists of same length: the former is the charactors of an original sentence, the latter is the corresponding BI tags.
        Notes: there is no tag "O" in the task of word segmentation.
    '''

    if prefix == 'train':
        path = '../Data/trainset/train_cws.txt'
    elif prefix == 'dev':
        path = '../Data/devset/val_cws.txt'
    elif prefix == 'test':
        path = '../Data/testset1/test_cws.txt'
    else:
        raise FileNotFoundError('Prefix should be "train", "dev" or "test"')
    
    #if os is Windows
    if (False): 
        path.replace('/', '\\', 10)
    
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    samples = []
    count = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            '''
            if count == batch_size:
                batches.append(samples)
                samples = []
                count = 0'''
            charactors = []
            tags = []
            line = line.strip().split()
            for word in line:
                for i, c in enumerate(word):
                    charactors.append(c)
                    if i == 0:
                        tags.append('B')
                    else:
                        tags.append('I')
            samples.append((charactors, tags))
            count += 1
    
    while True:
        yield random.sample(samples, batch_size)
    

#unit test
if __name__ == '__main__':
    TEST_NUM = 2
    g = wordseg_sample_generator(2, 'train')
    count = 0
    for i in g:
        print(i)
        count += 1
        if count == TEST_NUM:
            break