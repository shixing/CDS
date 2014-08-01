import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


class Analysis:
    
    def get_hashkey_from_count(self,lsh,count):
        # return hashkey and buckets
        neles = []
        engine = lsh.engine2
        for key in engine.storage.buckets['rbp']:
            neles.append(len(engine.storage.buckets['rbp'][key]))
        idx = neles.index(count)
        keys = engine.storage.buckets['rbp'].keys()
        return keys[idx],engine.storage.buckets['rbp'][keys[idx]]

    def show_dist(self,neles,num_bins):
        n, bins, patches = plt.hist(neles, num_bins, facecolor = 'green')
        plt.show()

    def show_cos_hist(self,vecs):
        if type(vecs) == list:
            vecs = np.array(vecs)
        dt = np.dot(vecs, vecs.T)
        norm = np.sqrt(np.sum(vecs * vecs,axis = 1))
        norm = norm.reshape(1,-1)
        denom = np.dot(norm.T,norm)
        cos_matrix = dt/denom
        plt.hist(cos_matrix.reshape(-1),20)
        plt.show()
        return cos_matrix
