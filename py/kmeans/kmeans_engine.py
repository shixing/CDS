import numpy as np
from kmeans.kmeans_restart import kmeans_hamming_restart

class KmeansEngine:
    def __init__(self,engine,k,num_restart):
        self.k = k
        self.num_restart = num_restart
        self.engine = engine
        self.buckets = self.engine.storage.buckets['rbp']
        self.cluster(k,num_restart)
        self.build_representative_vectors()

    def cluster(self,k,num_restart):
        keys = self.engine.storage.buckets['rbp'].keys()
        khr = kmeans_hamming_restart(keys,k,num_restart)
        self.centroids = khr[0]
        self.keys = keys
        self.keys_buckets = [[]*len(self.centroids)]
        for kidx,cidx in enumerate(khr[1]):
            self.keys_buckets[cidx].append(self.keys[kidx])
        
    def build_representative_vectors(self):
        self.r_vectors = []
        for cidx,keys_bucket in enumerate(self.keys_buckets):
            vector_list = []
            for key in keys_bucket:
                vector_list.extend(self.buckets[key])
            mean_vector = np.mean(vector_list,axis = 0)
            self.r_vectors.append(mean_vector)
    
    
        
