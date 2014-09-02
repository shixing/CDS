import os
import sys
import logging
import cPickle

import gensim
import configparser
import numpy as np

from vector.word2vec import MyWord2Vec
from vector.composition import Composition
from utils.config import get_config
from utils.heap import FixSizeHeap
from corpus.wordCollect import WordList

class Space:
    # the base class to have:
    # matrix1 : need to save
    # matrix2 : need to save
    # wordlist1
    # wordlist2
    # w2v
    # composition_matrix
    # dimension

    # All the configuration is in config file
    # Other searching class should inherit this class

    def get_transfer_vec(self,word,idx):
        # idx = 0 or 1
        index = self.wordlist[idx].index[word]
        return self.matrix[idx][index]

    def get_transfer_vec1(self,word):
        index = self.wordlist1.index[word]
        return self.matrix1[index]

    def get_transfer_vec2(self,word):
        index = self.wordlist2.index[word]
        return self.matrix2[index]

    def load_space_from_config(self,config):
        self._load_external_variable(config)
        
        fq = config.getint('space','filter_frequency')
        if fq == 1:
            self._filter_wordlist(config)

        self._build_matrix(config)
        self._add_index_to_matrix()
    
    def build_space_from_config(self,config):
        self._load_external_variable(config)

        fq = config.getint('space','filter_frequency')
        if fq == 1:
            self._filter_wordlist(config)

        self._build_matrix(config)
        self._add_index_to_matrix()
    
    def save_space_from_config(self,config):
        self._save_matrix(config)

    def _add_index_to_matrix(self):
        self.wordlist = [self.wordlist1,self.wordlist2]
        self.matrix = [self.matrix1,self.matrix2]

    def _load_external_variable(self,config):
        w2v = MyWord2Vec()
        w2v.load(config)
        wordlist1_path = config.get('path','adj_words')
        wordlist1 = WordList()
        wordlist1.load(wordlist1_path)
        wordlist1.build_index()
        wordlist2_path = config.get('path','noun_words')
        wordlist2 = WordList()
        wordlist2.load(wordlist2_path)
        wordlist2.build_index()
        com = Composition(config)
        com.load_1()

        self.w2v = w2v
        self.wordlist1 = wordlist1
        self.wordlist2 = wordlist2
        self.composition_matrix = com.wrm
        self.dimension = self.w2v.model.layer1_size

    def _save_matrix(self,config):
        logging.info("Saving Space")
        path = config.get('space','model_path')
        # matirx1 matrix2 kd_side kd_tree
        np.save(path+'.matrix1.npy',self.matrix1)
        np.save(path+'.matrix2.npy',self.matrix2)

    def _load_matrix(self,config):
        path = config.get('space','model_path')
        logging.info("Loading Space: Matrix1")
        self.matrix1 = np.load(path + '.matrix1.npy')
        logging.info("Loading Space: Matrix2")
        self.matrix2 = np.load(path + '.matrix2.npy')

    def _filter_wordlist(self,config):
        topn = config.getint('space','topn')
        logging.info('Filter wordlist1 to get top {} words'.format(topn))
        self.wordlist1.filter_frequency(self.w2v,topn)
        self.wordlist1.build_index()
        logging.info('Filter wordlist2 to get top {} words'.format(topn))
        self.wordlist2.filter_frequency(self.w2v,topn)
        self.wordlist2.build_index()

    def _build_matrix(self,config):
        s = np.shape(self.composition_matrix)
        half = s[1] / 2
        # matrix that has already transfered
        logging.info("Building Space: Building Matrix1")
        self.matrix1 = self._list2matrix(self.wordlist1,self.w2v,self.composition_matrix[:,:half])
        logging.info("Building Space: Building Matrix2")
        self.matrix2 = self._list2matrix(self.wordlist2,self.w2v,self.composition_matrix[:,half:])
        assert(np.shape(self.matrix1)[0] == len(self.wordlist1.words))
        assert(np.shape(self.matrix2)[0] == len(self.wordlist2.words))
        
    def _list2matrix(self,wordlist,word_vector,composition_matrix):
        matrix = np.zeros((len(wordlist.words),word_vector.model.layer1_size))
        i = 0
        for word in wordlist.words:
            vector = word_vector.getNorm(word)
            matrix[i] = vector
            i+=1
        matrix = np.dot(composition_matrix,matrix.T)
        matrix = matrix.T
        return matrix
    
    # some basic computation
    def query1_naive_matrix(self,vec,matrix):
        dt = np.dot(matrix,vec.T)
        denom = np.linalg.norm(vec,2)*np.sqrt(np.sum(matrix * matrix, axis = 1))
        dists = dt / denom
        topn = np.argsort(dists)[::-1]
        return topn,dists
        
    def query1_naive(self,vec,idx):
        # search the nearest word in idx-th matrix
        # return the topn list, the index
        dt = np.dot(self.matrix[idx],vec.T)
        denom = np.linalg.norm(vec,2)*np.sqrt(np.sum(self.matrix[idx] * self.matrix[idx], axis = 1))
        dists = dt / denom
        topn = np.argsort(dists)[::-1]
        return topn,dists
        
    def compose2(self,w1,w2):
        # compose two words
        if w1 in self.wordlist1.index and w2 in self.wordlist2.index:
            v1 = self.get_transfer_vec1(w1)
            v2 = self.get_transfer_vec2(w2)
            return v1 + v2
        else:
            return None
    
    

    
def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_path = sys.argv[1]
    config = get_config(config_path)
    space = Space()
    space.build_space_from_config(config)
    space.save_space_from_config(config)


if __name__=="__main__":
    main()
