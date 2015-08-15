import os
import sys
import logging
import cPickle
import random

import gensim
import configparser
import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import RandomDiscretizedProjections
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.distances.angular import AngularDistance
from bisect import bisect_left
from bitarray import bitarray
from nearpy.storage import MemoryStorage
from vector.engine2 import Permutation
from utils.distance import hamming_distance,cosine_distance


class DoubleEngine:

    def _build_rdp_engine(self,matrix,rdp,normals):
        # Dimension of our vector space
        dimension = np.shape(matrix)[1]
        n = np.shape(matrix)[0]
        # Create a random binary hash with 10 bits

        # Create engine with pipeline configuration
        engine = Engine(dimension, lshashes=[rdp],storage = MemoryStorage())
        rdp.vectors = normals

        for index in range(n):
            v = matrix[index]
            engine.store_vector(v, '%d' % index)
            
        return engine
    
        

    def process2(self,vectors1,vectors2,num_bit,bin_width):
        
        # build engine
        self.dimension = np.shape(vectors1)[1]
        self.rdp = RandomDiscretizedProjections('rdp',num_bit,bin_width)
        self.rbp = RandomBinaryProjections('rbp',num_bit)
        self.rdp.reset(self.dimension)
        self.rbp.reset(self.dimension)
        self.normals = self.rdp.vectors
        self.rbp.normals = self.normals
        self.engine1 = self._build_rdp_engine(vectors1,self.rdp,self.normals)
        self.engine2 = self._build_rdp_engine(vectors2,self.rdp,self.normals)
        
        # create new key
        buckets1 = self.engine1.storage.buckets['rdp']
        buckets2 = self.engine2.storage.buckets['rdp']
        
        self.rbdp = {}

        print 'len of buckets1', len(buckets1)
        print 'len of buckets2', len(buckets2)

        keys_int1 = []
        keys_int2 = []

        for key in buckets1:
            ks = [int(x) for x in key.split('_')]
            keys_int1.append(ks)

        for key in buckets2:
            ks = [int(x) for x in key.split('_')]
            keys_int2.append(ks)

        for idx1,key1 in enumerate(buckets1):
            if idx1 % 100 == 0:
                logging.info('{} {}/{}'.format(key1,idx1,len(buckets1)))
            for idx2,key2 in enumerate(buckets2):
                ks1 = keys_int1[idx1]
                ks2 = keys_int2[idx2]
                new_key = [ks1[i] + ks2[i] for i in xrange(len(ks1))]
                new_key = ''.join(['1' if x>=0 else '0' for x in new_key])
                if not new_key in self.rbdp:
                    self.rbdp[new_key] = []
                self.rbdp[new_key].append((key1,key2))
        
    def build_permute_index(self,num_permutation,beam_size,hamming_beam_size):
        self.num_permutation = num_permutation
        self.hamming_beam_size = hamming_beam_size
        self.beam_size = beam_size
        self.projection_count = self.rbp.projection_count
        
        # add permutations
        self.permutations = []
        for i in xrange(self.num_permutation):
            p = Permutation(self.projection_count)
            self.permutations.append(p)

        # convert current buckets to an array of bitarray
        buckets = self.rbdp
        original_keys = []
        for key in buckets:
            ba = bitarray(key)
            original_keys.append(ba)

        # build permutation lists
        self.permuted_lists = []
        i = 0
        for p in self.permutations:
            logging.info('Creating Permutation Index: #{}/{}'.format(i,len(self.permutations)))
            i+=1
            permuted_list = []
            for ba in original_keys:
                c = ba.copy()
                p.permute(c)
                permuted_list.append((c,ba))
            # sort the list
            permuted_list = sorted(permuted_list)
            self.permuted_lists.append(permuted_list)
        

    def get_neighbour_keys(self,bucket_key,k):
        # O( np*beam*log(np*beam) )
        # np = number of permutations
        # beam = self.beam_size
        # np * beam == 200 * 100 Still really fast

        query_key = bitarray(bucket_key)
        topk = set()
        for i in xrange(len(self.permutations)):
            p = self.permutations[i]
            plist = self.permuted_lists[i]
            candidates = p.search_revert(plist,query_key,self.beam_size)
            topk = topk.union(set(candidates))
        topk = list(topk)
        topk = sorted(topk, key = lambda x : hamming_distance(x,query_key))
        topk_bin = [x.to01() for x in topk[:k]]
        return topk_bin

    def n2(self,key1,key2,v):
        #return [(cos_dist,(idx1,idx2))]
        def matrix_list(engine,key):
            # return a matrix and a list of keys
            items = engine.storage.buckets['rdp'][key]
            m = []
            l = []
            for v,key in items:
                m.append(v)
                l.append(int(key))
            m = np.array(m)    
            return m,l
        m1,l1 = matrix_list(self.engine1,key1)
        m2,l2 = matrix_list(self.engine2,key2)
        len1 = len(l1)
        len2 = len(l2)
        # a . v 
        av = np.dot(m1,v)
        av = np.repeat(av,len2).reshape(len1,len2)
        # b . v
        bv = np.dot(m2,v)
        bv = np.repeat(bv,len1).reshape(len2,len1).T
        # nominator = a.v + b.v
        nomi = av + bv
        # |v|
        nv = np.linalg.norm(v,2)
        # a.a
        aa = np.sum(m1*m1,axis = 1)
        aa = np.repeat(aa,len2).reshape(len1,len2)
        # b.b
        bb = np.sum(m2*m2,axis = 1)
        bb = np.repeat(bb,len1).reshape(len2,len1).T
        # a.b
        ab = np.dot(m1,m2.T)
        # denominator 
        deno = np.sqrt(aa + bb + 2 * ab) * nv
        # distance matrix 
        dism = nomi / deno
        dist = []
        for i in xrange(len1):
            for j in xrange(len2):
                dis = dism[i,j]
                dist.append((dis,(l1[i],l2[j])))
        return dist

    def neighbours2(self,v,n):
        # one important assumption: just have one hash method
        # Collect candidates from all buckets from all hashes
        candidates = []
        direct_bucket_keys = self.rbp.hash_vector(v)

        # Get the neighbours of candidate_bucket_keys
        candidate_bucket_keys = []
        
        for bucket_key in direct_bucket_keys:
            neighbour_keys = self.get_neighbour_keys(bucket_key,self.hamming_beam_size)
            candidate_bucket_keys.extend(neighbour_keys)
        
        dists = []
        for bucket_key in candidate_bucket_keys:
            comb = self.rbdp[bucket_key]
            print bucket_key, len(comb)
            for key1,key2 in comb:
                dist = self.n2(key1,key2,v)
                dists.extend(dist)

        dists = sorted(dists,key = lambda x: -x[0])
        return dists[:n]
        # If there is no vector filter, just return list of candidates
        return dists

def test():
    
    import numpy as np
    import logging
    from vector.engine3 import DoubleEngine
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    engine = DoubleEngine()
    m1 = np.random.rand(10000,2)
    m2 = np.random.rand(10000,2)
    v = np.random.rand(2)
    engine.process2(m1,m2,15,0.1)
    engine.build_permute_index(5,6,5)
    print engine.neighbours2(v,100)

if __name__ == '__main__':
    test()
