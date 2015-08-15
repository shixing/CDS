import sys
import numpy as np
from vector.LSH import LSH
from corpus.wordCollect import WordList
from utils.config import get_config
import logging
from nearpy.hashes import RandomBinaryProjections
from utils.heap import FixSizeHeap
from vector.word2vec import MyWord2Vec
from vector import objectives
from vector.engine3 import DoubleEngine

def build_environment(config):
    lsh = LSH()
    lsh._load_external_variable(config)
    lsh._build_matrix(config)
    w2v = lsh.w2v

    adjs = lsh.wordlist1
    nouns = lsh.wordlist2
    adjm = lsh.matrix1
    nounm = lsh.matrix2
    
    # build engine
    num_bits = 15
    bin_width = 0.1
    engine = DoubleEngine()
    engine.process2(adjm,nounm,num_bit,bin_width)
    
    num_permutation = 50
    beam_size = 50
    num_neighbour = 100
    engine.build_permute_index(num_permutation,beam_size,num_neighbour)
    
    return lsh,w2v,engine,adjm,adjs,nounm,nouns

def query_word1(word,n,lsh,engine,adjs,nouns):
    v = lsh.w2v.getNorm(word)
    query(v,n,engine,adjs,nouns)

def query_word2(word1,word2,n,lsh,engine,adjs,nouns):
    v = lsh.compose2(word1,word2)
    query(v,n,engine,adjs,nouns)

def query(v,n,engine,adjs,nouns):
    dists = engine.neighbours2(v,n)
    for dis,idxs in dists:
        adj = adjs[idxs[0]]
        noun = nouns[idxs[1]]
        print adj,noun,dis
    


