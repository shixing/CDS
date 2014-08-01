import os
import sys
import logging
import cPickle

import gensim
import configparser
import numpy as np
import scipy
import scipy.spatial
from scipy.spatial import kdtree
# patch module-level attribute to enable pickle to work
kdtree.node = kdtree.KDTree.node
kdtree.leafnode = kdtree.KDTree.leafnode
kdtree.innernode = kdtree.KDTree.innernode

from utils.config import get_config
from utils.heap import FixSizeHeap
from corpus.wordCollect import WordList
from vector.word2vec import MyWord2Vec
from vector.composition import Composition
from vector.space import Space

class BruteForceSearch(Space):
    # need to run Space first to have the matrix1 and matrix2 ready
    # Building the KD-tree is much faster than load from pickle
    
    def load_from_space(self,space):
        self.matrix1 = space.matrix1
        self.matrix2 = space.matrix2
        self.wordlist1 = space.wordlist1
        self.wordlist2 = space.wordlist2
        self.w2v = space.w2v
        self.composition_matrix = space.composition_matrix
        self.__build_kd_tree()

    def load_from_config(self,config):
        self.load_space_from_config(config)
        self.__build_kd_tree()

    def __build_kd_tree(self):
        logging.info("Building BruteForce Model")
        # set kd_side
        self.kd_side = 1
        if np.shape(self.matrix1)[0] < np.shape(self.matrix2)[0]:
            self.kd_side = 2
        s = np.shape(self.composition_matrix)
        half = s[1] / 2
        # matrix that has already transfered
        logging.info("Building BruteForce Model: Building KDTree")
        if self.kd_side == 1:
            self.kd_tree = scipy.spatial.KDTree(self.matrix1,np.shape(self.matrix1)[0]+1)
        elif self.kd_side == 2:
            self.kd_tree = scipy.spatial.KDTree(self.matrix2,np.shape(self.matrix2)[0]+1)
        logging.info("Building BruteForce Model Done")
   
    def query2_bf(self,query_word,k):
        # pure bruteforce, without KD stree
        logging.info("Quering word: {}".format(query_word))
        query_vector = self.w2v.getNorm(query_word)
        if query_vector == None:
            return None
        heap = FixSizeHeap(k)
        for i in xrange(np.shape(self.matrix1)[0]):
            for j in xrange(np.shape(self.matrix2)[0]):
                if i % 10 == 0 and j == 0:
                    logging.info("Searched {}/{} words".format(i,len(self.wordlist2.words)))
                v1 = self.matrix1[i]
                v2 = self.matrix2[j]
                vq = v1 + v2
                dis = np.linalg.norm(vq-query_vector,2)
                item = (-dis,(i,j))
                heap.push(item)

        results = [(-x[0],( self.wordlist1[x[1][0]], self.wordlist2[x[1][1]] ) ) for x in heap.data]
        results = sorted(results)
        return results


    def query2(self,query_word,k):
        # return 2 decomposed words with kd_tree
        logging.info("Quering word: {}".format(query_word))
        query_vector = self.w2v.getNorm(query_word)
        if query_vector == None:
            return None
        loop_matrix = None
        heap = FixSizeHeap(k)

        pre_dis = float("inf")
        
        if self.kd_side == 1:
            loop_matrix = self.matrix2
        elif self.kd_side == 2:
            loop_matrix = self.matrix1
        for i in xrange(np.shape(loop_matrix)[0]):
            if i % 10 == 0:
                logging.info("Searched {}/{} words".format(i,len(self.wordlist2.words)))
            v1 = loop_matrix[i,:]
            v2 = query_vector - v1
            # single NN
            dis, v2_index = self.kd_tree.query(v2,k=1,distance_upper_bound=pre_dis)
            item = None
            if self.kd_side == 1:
                item = (-dis,(v2_index,i))
            elif self.kd_side == 2:
                item = (-dis,(i,v2_index))
            heap.push(item)
            pre_dis = -heap.data[0][0]

        results = [(-x[0],( self.wordlist1[x[1][0]], self.wordlist2[x[1][1]] ) ) for x in heap.data]
        results = sorted(results)
        return results


def test():
    # to run this test, need to have remove the __init__() of BruteForceSearch()
    bf = BruteForceSearch()
    matrix1 = np.random.random((2,2))
    matrix2 = np.random.random((2,2))
    bf.kd_side = 2
    bf.matrix1 = matrix1
    bf.matrix2 = matrix2
    bf.wordlist1 = range(0,2);
    bf.wordlist2 = range(2,4);
    bf.kd_tree = scipy.spatial.KDTree(matrix2)
    query = matrix1[0] + matrix2[0]
    print bf.matrix1
    print bf.matrix2
    print query
    print bf.search2(query)
    
    # test_save
    bf.save('/Users/xingshi/Workspace/misc/CDS/var/kdtree_test')


def main_build():
    # build for wiki
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_path = sys.argv[1]
    config = get_config(config_path)
    bf = BruteForceSearch()
    bf.build_from_config(config)
    bf.save_from_config(config)


def main_query():
    # build for wiki
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_path = sys.argv[1]
    config = get_config(config_path)
    bf = BruteForceSearch()
    bf.build_from_config(config)
    results = bf.query2_bf("high_school",10)
    print results




if __name__=="__main__":
    main_query()
