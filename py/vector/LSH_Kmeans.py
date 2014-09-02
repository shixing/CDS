# Unfinished, aborted code

import gensim
import configparser
import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import RandomDiscretizedProjections
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.distances.angular import AngularDistance
from nearpy.storage import MemoryStorage

from vector.LSH import LSH
from vector.engine import PermuteEngine
from corpus.wordCollect import WordList
from kmeans.kmeans_engine import KmeansEngine


class LSH_Kmeans(LSH):
    # LSH_Kmeans have the following new features:
    # wordlist1 (filtered top 20k)
    # wordlist2 (filtered top 20k)
    # matrix1
    # matrix2

    

    def load_from_config_light(self,config,debug = False):
        self._load_external_variable(config)
        self._filter_wordlist(config)
        self._build_matrix()        
 
    def build_index_kmeans(k,num_bits):
        # hash the original vector in matrxi1 and matrix2 into engine1 and engine2
        engine1 = self._build_rbp_engine(self.matrix1,num_bits)
        engine2 = self._build_rbp_engine(self.matrix2,num_bits)

        # cluster the buckets into k clusters
        
        kmeans_engine1 = KmeansEngine(engine1,k,20)
        kmeans_engine2 = KmeansEngine(engine2,k,20)

        # two for-loop to get k^2 sum_vectors and hash the new sum_vectors into engine_sum

        rbp = RandomBinaryProjections('rbp',num_bit)
        dimension = np.shape(self.matrix1)[1]
        engine_sum = PermuteEngine(dimension,lshashes[rbp],storage = MemoryStorage())
        matrix_sum = np.zeros((k*k,dimension))
        i = 0
        for i1,r1 in enumerate(kmeans_engine1.r_vectors):
            for i2,r2 in enumerate(kmeans_engine2.r_vectors):
                s = r1 + r2
                matrix_sum[i] = s
                i+=1
                key = '{}_{}'.format(i1,i2)
                engine_sum.store_vector(s,key)
        
        self.kmeans_engine1 = kmeans_engine1
        self.kmeans_engine2 = kmeans_engine2
        self.engine_sum = engine_sum
        self,matrix_sum - matrix_sum
        
    def query_1_2_v(self,query_vector,k,naive = False):
        
        if query_vector == None:
            return None
        
        topn1 = []
        topn2 = []
        topn_dists = []
        if naive:
            # naive methods
            topn,dists = self.query1_naive_matrix(query_vector,self.matrix_sum)
            num_cluster = int(np.sqrt(len(self.matrix_sum)))
            topn1 = [x/num_cluster for x in topn]
            topn2 = [x%num_cluster for x in topn]
            topn_dists = dists[topn[:k]]
        
        else:
            topk = self.engine_sum.neighbours2(query_vector)
            for x in tok[:k]:
                dis = x[2]
                topn_dists.append(dis)
                idx_string = x[1]
                idx_s = idx_string.split('_')
                idx1 = int(idx_s[0])
                idx2 = int(idx_s[1])
                topn1.append(idx1)
                topn2.append(idx2)
                
        
            
