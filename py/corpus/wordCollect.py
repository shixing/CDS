import sys
import os
import logging

import configparser

from utils.config import get_config
from vector.word2vec import MyWord2Vec


class WordList:
    
    def build_index(self):
        self.index = {}
        for i,word in enumerate(self.words):
            self.index[word] = i

    def filter(self,w2v):
        new_words = []
        for word in self.words:
            if word in w2v.model.vocab:
                new_words.append(word)
        self.words = new_words

    def filter_frequency(self,w2v,topn):
        def get_count(word):
            vocab = w2v.model.vocab[word]
            return vocab.count
        new_words = []
        for word in self.words:
            count = get_count(word)
            new_words.append((-count,word))
        new_words = sorted(new_words)
        new_words = new_words[:topn]
        new_words = [x[1] for x in new_words]
        self.words = new_words

    def load(self,fn):
        self.words = []
        for line in open(fn):
            word = line.strip()
            self.words.append(word)
    
    def collect(self,taglist,fn,fn_pos):
        self.words = []
        tagset = set(taglist)
        d = {}
        f = open(fn)
        fpos = open(fn_pos)
        k = 0
        while True:
            k += 1
            if k % 10000 == 0:
                logging.info("Processed {} lines".format(k))
            line = f.readline()
            pos_line = fpos.readline()
            if not line:
                break
            words = line.strip().split()
            poses = pos_line.strip().split()
            for i in xrange(len(poses)):
               pos = poses[i] 
               if pos in tagset:
                   word = words[i]
                   if not word in d:
                       d[word] = 1
                       self.words.append(word)
    
    def save(self,fn):
        f = open(fn,"w")
        for word in self.words:
            f.write(word+"\n")
        f.close()

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_fn = sys.argv[1]
    config = get_config(config_fn)
    ftext = config.get('path','short_abstracts_text')
    fpos = config.get('path','short_abstracts_pos')
    fn_noun = config.get('path','noun_words')
    fn_adj = config.get('path','adj_words')

    w2v = MyWord2Vec()
    w2v.load(config)

    #collect Noun Words
    nounList = WordList()
    nounList.load(fn_noun)
    nounList.filter(w2v)
    #nounList.collect(['NN','NNS'],ftext,fpos)
    nounList.save(fn_noun)
    #collect Adj Words
    adjList = WordList()
    adjList.load(fn_adj)
    adjList.filter(w2v)
    #adjList.collect(['JJ','JJS','JJR'],ftext,fpos)
    adjList.save(fn_adj)


    

if __name__ == "__main__":
    main()
