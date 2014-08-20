import os
import sys
import logging
import cPickle

import gensim
import configparser
import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import RandomDiscretizedProjections
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.distances.angular import AngularDistance

from vector.LSH import LSH
from vector.engine import PermuteEngine
from corpus.wordCollect import WordList

class LSH2gram(LSH):
    # the LSH2gram have the following new additional features:
    # wordlist_2gram: A_B
    # matrix_2gram:
    # engine_2gram:

    def load_from_config_light(self,config, debug = False):
        # just load the matrix
        self._load_external_variable(config)
        
        self._load_matrix(config)

        self._load_wordlist_2gram(config)

        

        logging.info('LSH2gram: Loading matrix_2gram')
        fn_matrix_2gram = config.get('lsh2gram','matrix_2gram')
        self.matrix_2gram = np.load(fn_matrix_2gram)

        logging.info('LSH2gram: Building Engine_2gram')
        # build the index
        num_bit = config.getint('lsh2gram','num_bit')

        self._build_index_2gram(num_bit)
        self._add_index_to_matrix()


    def build_from_config(self,config):
        self._load_external_variable(config)
        
        self._load_matrix(config)

        self._load_wordlist_2gram(config)
        
        self._build_matrix_2gram(config)

        logging.info('LSH2gram: Building Engine_2gram')
        # build the index
        num_bit = config.getint('LSH2gram','num_bit')

        self._build_index_2gram(num_bit)
        self._add_index_to_matrix()
    
        

    def save_from_config(self,config):
        #save:
        # wordlist_2gram
        # matrix_2gram
        logging.info('Saving matrix_2gram and wordlist_2gram')
        fn = config.get('path','bigram_adj_noun_final')
        self.wordlist_2gram.save(fn)
        fn_matrix_2gram = config.get('lsh2gram','matrix_2gram')
        np.save(fn_matrix_2gram,self.matrix_2gram)

    def _add_index_to_matrix(self):
        self.matrix = [self.matrix_2gram]
        self.wordlist = [self.wordlist_2gram]


    def _get_engine(self):
        if hasattr(self,'engine_2gram'):
            return self.engine_2gram
        else:
            return None
        

    def _build_matrix_2gram(self,config):
        logging.info('LSH2gram: Building Matrix_2gram')
        # build matrix_2gram
        n = len(self.wordlist_2gram.words)
        d = self.dimension
        uvm = np.zeros((n,d*2))
        new_words = []
        i = 0
        for word in self.wordlist_2gram.words:
            ww = word.lower().split()
            assert(len(ww) == 2)
            u = self.w2v.getNorm(ww[0])
            v = self.w2v.getNorm(ww[1])
            if u == None or v == None:
                continue
            uvm[i,:d] = u
            uvm[i,d:] = v
            i += 1
            new_words.append(ww[0]+'_'+ww[1])
        self.wordlist_2gram.words = new_words
        uvm = uvm[:i,:]
        self.matrix_2gram = np.dot(self.composition_matrix, uvm.T).T
        

    def _load_wordlist_2gram(self,config):
        logging.info('LSH2gram: Loading wordlist_2gram')
        # laod the 2grams
        f2gram = config.get('path','bigram_adj_noun')
        self.wordlist_2gram = WordList()
        self.wordlist_2gram.load(f2gram)
        self.wordlist_2gram.build_index()


    def _build_index_2gram(self,num_bit):
        self.engine_2gram = self._build_rbp_engine(self.matrix_2gram,num_bit)


    def query_2_2(self,qw1,qw2,k,naive = False):
        logging.info("Quering words: {} {}".format(qw1,qw2))
        query_vector = self.compose2(qw1,qw2)
        return self.query_1_2_v(query_vector,k,naive)

    def query_1_2(self,query_word,k,naive = False):
        logging.info("Quering word: {}".format(query_word))
        query_vector = self.w2v.getNorm(query_word)
        return self.query_1_2_v(query_vector,k,naive)

    def query_1_2_v(self,query_vector,k,naive = False):
        # input 1 word
        # return 2 decomposed words
        # [(dis,(adj,noun))]

        if query_vector == None:
            return None
        
        if naive:
            topn,dists = self.query1_naive(query_vector,0)
            words = np.array(self.wordlist[0].words)
            topn_words = words[topn[:k]]
            topn_dists = dists[topn[:k]]
            return zip(topn_dists,topn_words)

        nnf = NearestFilter(k)
        self.engine_2gram.vector_filters = [nnf]
        self.engine_2gram.distance = AngularDistance()

        topk = self.engine_2gram.neighbours2(query_vector)
        
        results = []
        for x in topk[:k]:
            dis = x[2]
            idx = int(x[1])
            word = self.wordlist_2gram.words[idx]
            results.append((1-dis,word))
            
        return results
        
        
