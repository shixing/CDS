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

from utils.config import get_config
from utils.heap import FixSizeHeap
from corpus.wordCollect import WordList
from vector.word2vec import MyWord2Vec
from vector.composition import Composition
from vector.space import Space





class LSH(Space):
    # need to run Space first to have the matrix1 and matrix2 ready

    def load_from_config(self,config):
        self.load_space_from_config(config)
        build_option = config.getint('LSH','build_option')
        num_bit = config.getint('LSH','num_bit')
        self.build_option = build_option
        self._build_index(num_bit)

    def _build_index(self,num_bit):
        logging.info("Building LSH Model: Building Index")
        if self.build_option == 1:
            self.engine1 = self._build_rbp_engine(self.matrix1,num_bit)
        elif self.build_option == 2:
            self.engine2 = self._build_rbp_engine(self.matrix2,num_bit)
        logging.info("Building LSH Model: Building Index Done")
    
    def _build_rbp_engine(self,matrix,num_bit):
        engine = self._get_engine()
        if engine != None:
            engine.clean_all_buckets()
        # Dimension of our vector space
        dimension = np.shape(matrix)[1]
        n = np.shape(matrix)[0]
        # Create a random binary hash with 10 bits
        rbp = RandomBinaryProjections('rbp', num_bit)
        # Create engine with pipeline configuration
        engine = Engine(dimension, lshashes=[rbp])

        for index in range(n):
            v = matrix[index]
            engine.store_vector(v, '%d' % index)
            
        return engine
   
    def _get_engine(self):
        engine = None
        if self.build_option == 1:
            if hasattr(self,'engine1'):
                engine = self.engine1
        elif self.build_option == 2:
            if hasattr(self,'engine2'):
                engine = self.engine2
        return engine
 
    def engine_status(self):
        import matplotlib.mlab as mlab
        import matplotlib.pyplot as plt
        engine = self._get_engine()
        neles = []
        for key in engine.storage.buckets['rbp']:
            neles.append(len(engine.storage.buckets['rbp'][key]))
        num_bins = 40
        n, bins, patches = plt.hist(neles, num_bins, facecolor = 'green')
        plt.show()
        return neles



    def _build_rdp_engine(self,matrix,projection_count,bin_width):
        # Dimension of our vector space
        dimension = np.shape(matrix)[1]
        n = np.shape(matrix)[0]
        # Create a random binary hash with 10 bits
        rdp = RandomDiscretizedProjections('rdp',projection_count,bin_width)
        # Create engine with pipeline configuration
        engine = Engine(dimension, lshashes=[rdp])
        
        # Index 1000000 random vectors (set their data to a unique string)
        for index in range(n):
            v = matrix[index]
            engine.store_vector(v, '%d' % index)
        return engine

    def query1_word_rank(self,word,idx):
        vec = self.get_transfer_vec(word,idx)
        return self.query1_vec_rank(vec,idx)

    def query1_vec_rank(self,vec,idx):
        topn = self.query1_vec(vec)
        topn_gold, dists = self.query1_naive(vec,idx)
        res = []
        i = 0
        for item in topn:
            index = int(item[1])
            dist = item[2]
            rank = np.where(topn_gold == index)[0][0]
            res.append((i,rank,index,dist))
            i+=1
        return res


    def query1_vec(self,vec): 
        engine = None
        if self.build_option == 1:
            engine = self.engine1
        elif self.build_option == 2:
            engine = self.engine2
        
        nnf = NearestFilter(10)
        engine.vector_filters = [nnf]
        engine.distance = AngularDistance()
        topn = engine.neighbours(vec)

        return topn # (vector,data,distance)



    def query2_word(self,query_word,k):
        # return 2 decomposed words
        logging.info("Quering word: {}".format(query_word))

        query_vector = self.w2v.getNorm(query_word)
        if query_vector == None:
            return None
        
        loop_matrix = None
        engine = None
        heap = FixSizeHeap(k)
        
        if self.build_option == 1:
            loop_matrix = self.matrix2
            engine = self.engine1
        elif self.build_option == 2:
            loop_matrix = self.matrix1
            engine = self.engine2

        nnf = NearestFilter(1)
        engine.vector_filters = [nnf]
        engine.distance = AngularDistance()

        # find the query_word's index
        query_index = None
        ncol = np.shape(self.composition_matrix)[1]
        transformed_query_vector = np.dot(self.composition_matrix[:,ncol/2:],query_vector.T).T
        query_topk = engine.neighbours(transformed_query_vector)

        if len(query_topk) > 0:
            dist = query_topk[0][2]
            if dist < 0.001:
                query_index = int(query_topk[0][1])

        for i in xrange(np.shape(loop_matrix)[0]):
            if i % 1000 == 0:
                logging.info("Searched {}/{} words".format(i,len(self.wordlist1.words)))
            v1 = loop_matrix[i,:]
            v2 = query_vector - v1
            v1n = np.linalg.norm(v1,2)
            v2n = np.linalg.norm(v2,2)
            # filter1: the small vector
            if v1n < 0.1 or v2n < 0.1:
                continue
        
            # single NN
            topk = engine.neighbours(v2)
            if len(topk) == 0:
                continue
            
            for top in topk:
                dis = top[2]
                v2_index = int(top[1])
                # filter 2: same vector
                if v2_index == query_index:
                    continue
                item = None
                if self.build_option == 1:
                    item = (-dis,(v2_index,i))
                elif self.build_option == 2:
                    item = (-dis,(i,v2_index))
                heap.push(item)

        results = [(-x[0],( self.wordlist1.words[x[1][0]], self.wordlist2.words[x[1][1]] ) ) for x in heap.data]
        results = sorted(results)
        return results

def main_query():
    # build for wiki
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_path = sys.argv[1]
    config = get_config(config_path)
    lsh = LSH()
    lsh.load_from_config(config)
    results = lsh.query2_word("high_school",10)
    print results

if __name__=="__main__":
    main_query()
