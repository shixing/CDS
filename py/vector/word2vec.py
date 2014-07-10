import sys,os
import gensim, logging
import configparser
from corpus.mySentence import MySentences
from utils.config import *

class MyWord2Vec:

    def train(self,sentences,config):
        min_count = config.getint('word2vec','min_count')
        size = config.getint('word2vec','size')
        workers = config.getint('word2vec','workers')
        self.model = gensim.models.Word2Vec(min_count = min_count, size = size, workers = workers)
        self.model.build_vocab(sentences)
        self.model.train(sentences)
    
    def save(self,config):
        model_path = config.get('word2vec','model_path')
        self.model.save(model_path)
        
    def load(self,config):
        model_path = config.get('word2vec','model_path')
        self.model = gensim.models.Word2Vec.load(model_path)

    def evaluate_ABCD(self,config):
        # A is to B as C is to D
        question_words = config.get('path','question_words')
        self.model.accuracy(question_words)
        
def test():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_path = sys.argv[1]
    config = get_config(config_path)
    #sa = config.get('test','test_data'
    sa = config.get('path','short_abstracts')
    sentences = MySentences(sa)
    myw2v = MyWord2Vec()
    myw2v.train(sentences,config)
    myw2v.save(config)
    




if __name__ == '__main__':
    test()
