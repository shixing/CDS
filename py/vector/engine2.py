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

from utils.distance import hamming_distance,cosine_distance

class Permutation:
    # [n] -> [n] permutations

    def __init__(self,n):
        m = range(n)
        for end in xrange(n-1,0,-1):
            r = random.randint(0,end)
            tmp = m[end]
            m[end] = m[r]
            m[r] = tmp
        self.mapping = m
    
    def permute(self,ba): # inplace
        c = ba.copy()
        for i in xrange(len(self.mapping)):
            ba[i] = c[self.mapping[i]]
        return ba
        
    def revert(self,ba):
        c = ba.copy()
        for i in xrange(len(self.mapping)):
            ba[self.mapping[i]] = c[i]
        return ba

    def search_revert(self,bas,ba,beam_size):
        # return the original key
        pba = ba.copy()
        self.permute(pba)
        assert(beam_size % 2 == 0)
        half_beam = beam_size / 2
        idx = bisect_left(bas,(pba,ba))
        start = max(0,idx - half_beam)
        end = min(len(bas),idx+half_beam)
        res =  bas[start:end]
        res = [x[1] for x in res]
        return res


class PermuteEngine(Engine):
    # has following variables:
    # beam_size
    # hamming_beam_size
    # projection_count
    # permutations = [ Permutation ]
    # permuted_lists

    def build_permute_index(self,num_permutation,beam_size,hamming_beam_size):
        self.num_permutation = num_permutation
        self.hamming_beam_size = hamming_beam_size
        self.beam_size = beam_size
        self.projection_count = self.lshashes[0].projection_count
        
        # add permutations
        self.permutations = []
        for i in xrange(self.num_permutation):
            p = Permutation(self.projection_count)
            self.permutations.append(p)

        # convert current buckets to an array of bitarray
        buckets = self.storage.buckets['rbp']
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

    def neighbours2(self,v):
        # one important assumption: just have one hash method
        # Collect candidates from all buckets from all hashes
        candidates = []
        direct_bucket_keys = []
        for lshash in self.lshashes:
            for bucket_key in lshash.hash_vector(v):
                direct_bucket_keys.append(bucket_key)
        
        # Get the neighbours of candidate_bucket_keys
        candidate_bucket_keys = []
        
        for bucket_key in direct_bucket_keys:
            neighbour_keys = self.get_neighbour_keys(bucket_key,self.hamming_beam_size)
            candidate_bucket_keys.extend(neighbour_keys)
        
        hash_name = self.lshashes[0].hash_name
        for bucket_key in candidate_bucket_keys:
            bucket_content = self.storage.get_bucket(hash_name,
                                                         bucket_key)
            candidates.extend(bucket_content)

        # calculate cosine distance
        candidate_matrix = np.zeros((len(candidates),self.lshashes[0].dim))
        for i in xrange(len(candidates)):
            candidate_matrix[i] = candidates[i][0]
        npv = np.array(v)
        npv = npv.reshape((1,len(npv)))
        cos_dists = cosine_distance(candidate_matrix,npv)
        cos_dists = cos_dists.reshape((-1,))                        
            
        # Apply distance implementation if specified
        if self.distance:
            candidates = [(candidates[i][0], candidates[i][1], 1-cos_dists[i]) for i in xrange(len(candidates))]

        # Apply vector filters if specified and return filtered list
        if self.vector_filters:
            filter_input = candidates
            for vector_filter in self.vector_filters:
                filter_input = vector_filter.filter_vectors(filter_input)
            # Return output of last filter
            return filter_input

        # If there is no vector filter, just return list of candidates
        return candidates


def test():

    from nearpy.hashes import RandomBinaryProjections
    from nearpy.hashes import RandomDiscretizedProjections
    from nearpy.filters.nearestfilter import NearestFilter
    from nearpy.distances.angular import AngularDistance

    matrix = np.random.random((1000,200))
    query = np.random.random(200)
    
    dimension = 200
    num_bit = 10
    engine = PermuteEngine()

    rbp = RandomBinaryProjections('rbp', num_bit)
    # Create engine with pipeline configuration
    engine = PermuteEngine(dimension, lshashes=[rbp])
    engine.distance = AngularDistance()

    for index in range(n):
        v = matrix[index]
        engine.store_vector(v, '%d' % index)
        
    


if __name__ == '__name__':
    test()
        
