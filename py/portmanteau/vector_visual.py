import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys
from vector.LSH_sumbeam import LSH_sumbeam
from corpus.wordCollect import WordList
from utils.config import get_config
import logging
from nearpy.hashes import RandomBinaryProjections
from utils.heap import FixSizeHeap
from vector.word2vec import MyWord2Vec

def v(label_list,vectors,fn):
    n = len(label_list)
    plt.figure(1,figsize=(8,3*n))
    for i in xrange(len(label_list)):
        label = label_list[i]
        vector = vectors[i]
        plt.subplot(n,1,i+1)
        vs = len(vector)
        x = range(vs)
        plt.bar(x,vector)
        plt.title(label)
    plt.savefig(fn)


def test():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    config_fn = sys.argv[1]
    config = get_config(config_fn)
    
    w2v = MyWord2Vec()
    w2v.load(config)
    
    fn = '/Users/xingshi/Workspace/data/wordlist/antonyms.txt'

    label_list = []
    vectors = []
    for line in open(fn):
        ll = line.split()
        w1 = ll[0]
        w2 = ll[1]
        v1 = w2v.getNorm(w1)
        v2 = w2v.getNorm(w2)
        vector = v1 - v2
        label = w1+'_'+w2
        vectors.append(vector)
        label_list.append(label)

   
    
    v(label_list,vectors,'antonyms.pdf')



if __name__ == '__main__':
    test()

