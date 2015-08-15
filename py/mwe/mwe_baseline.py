import sys
import numpy as np
from vector.LSH_sumbeam import LSH_sumbeam
from corpus.wordCollect import WordList
from utils.config import get_config
import logging
from nearpy.hashes import RandomBinaryProjections
from utils.heap import FixSizeHeap
from vector.word2vec import MyWord2Vec
from vector import objectives


def build_environment(config):
    lsh = LSH_sumbeam()
    w2v = MyWord2Vec()
    w2v.load(config)
    lsh.w2v = w2v

    # combine top 20k noun and 20k adj into a single wordlist
    topn = config.getint('space','topn')
    words = w2v.model.vocab.keys()
    wordlist = WordList()
    wordlist.words = words
    wordlist.filter_frequency(w2v,topn)
    wordlist.build_index()

    # build a matrix
    matrix = lsh._list2matrix_w2v(wordlist,lsh.w2v)

    # build an engine
    dim = np.shape(matrix)[1]
    num_bits = 15
    rbp = RandomBinaryProjections('rbp', num_bits)
    rbp.reset(dim)    
    engine = lsh._build_rbp_permute_engine(matrix,rbp)
    num_permutation = 50
    beam_size = 50
    num_neighbour = 100
    engine.build_permute_index(num_permutation,beam_size,num_neighbour)

    return lsh,engine,matrix,wordlist

def query_1(w,lsh,matrix,wordlist,n):
    vec = lsh.w2v.getNorm(w)
    topn,dists = objectives.objective1(vec,matrix)
    data = []
    for i in xrange(n):
        idx = topn[i]
        dis = dists[idx]
        w4 = wordlist.words[idx]
        data.append((dis,w4))
    data = sorted(data, key = lambda x: -x[0])
    return data
    
