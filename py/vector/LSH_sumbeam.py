# a beam search method for searching the sum

import logging

import gensim
import configparser
import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import RandomDiscretizedProjections
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.distances.angular import AngularDistance
from nearpy.storage import MemoryStorage

from vector.LSH2gram import LSH2gram
from vector.engine2 import PermuteEngine
from corpus.wordCollect import WordList
from utils.distance import cosine_distance
from utils.heap import FixSizeHeap


class LSH_sumbeam(LSH2gram):
    # LSH_Kmeans have the following new features:
    # wordlist1 (filtered top 20k)
    # wordlist2 (filtered top 20k)
    # matrix1
    # matrix2
    # engine1
    # engine2
    # rbp
    # dim

    def load_from_config_light(self,config,debug = False):
        self._load_external_variable(config)
        self._filter_wordlist(config)
        self._build_matrix(config)        
        
    def build_index_sumbeam(self,num_bits):
        # hash the original vector in matrxi1 and matrix2 into engine1 and engine2
        self.dim = np.shape(self.matrix1)[1]
        rbp = RandomBinaryProjections('rbp', num_bits)
        rbp.reset(self.dim)
        self.rbp = rbp
    
        engine1 = self._build_rbp_permute_engine(self.matrix1,rbp)
        engine2 = self._build_rbp_permute_engine(self.matrix2,rbp)
        self.engine1 = engine1
        self.engine2 = engine2

    def build_permute_index(self,num_permutation,beam_size,num_neighbour):
        self.engine1.build_permute_index(num_permutation,beam_size,num_neighbour)
        self.engine2.build_permute_index(num_permutation,beam_size,num_neighbour)
        
    def query_2_2(self,qw1,qw2,k,num_neighbour,naive = False):
        logging.info("Quering words: {} {}".format(qw1,qw2))
        query_vector = self.compose2(qw1,qw2)
        return self.query_1_2_v(query_vector,num_neighbour,k,naive)

    def query_1_2(self,query_word,k,num_neighbour,naive = False):
        logging.info("Quering word: {}".format(query_word))
        query_vector = self.w2v.getNorm(query_word)
        return self.query_1_2_v(query_vector,num_neighbour,k,naive)

    def query_1_2_v(self,query_vector,num_neighbour,k,naive):
        
        query_matrix = query_vector.reshape((1,-1))
        query_key = self.rbp.hash_vector(query_vector)[0]
        
        neighbour_keys_1 = self.engine1.get_neighbour_keys(query_key,num_neighbour)
        neighbour_keys_2 = self.engine2.get_neighbour_keys(query_key,num_neighbour)

        heap = FixSizeHeap(k)

        for i1,nkey1 in enumerate(neighbour_keys_1):
            for i2,nkey2 in enumerate(neighbour_keys_2):
                #logging.info("{} {} / {} {}".format(i1,i2,len(neighbour_keys_1),len(neighbour_keys_2)))
                vs1 = self.engine1.storage.get_bucket('rbp',nkey1)
                vs2 = self.engine2.storage.get_bucket('rbp',nkey2)
                n = len(vs1) * len(vs2)
                sum_matrix = np.zeros((n,self.dim))
                i = 0
                for v1,d1 in vs1:
                    for v2,d2 in vs2:
                        s = v1 + v2
                        sum_matrix[i] = s
                        i += 1
                cos_matrix = cosine_distance(sum_matrix,query_matrix)
                cos_matrix = cos_matrix.reshape((-1))
                topn = np.argsort(cos_matrix)[::-1]
                for i in topn:
                    dist = cos_matrix[i]
                    idx1 = i / len(vs2)
                    idx2 = i % len(vs2)
                    w1 = int(vs1[idx1][1])
                    w2 = int(vs2[idx2][1])
                    heap.push((dist,(w1,w2)))

        results = [(x[0],( self.wordlist1.words[x[1][0]], self.wordlist2.words[x[1][1]] ) ) for x in heap.data]
        results = sorted(results)
        return results
 
                    
                
                
