import os
import sys
import logging
import cPickle

import configparser
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



from utils.config import get_config
from vector.LSH import LSH
from vector.composition import Composition 

class ExperientInterface:

    def __init__(self,config):
        self.config = config;

    def run(self):
        raise NotImplementedError;
    
    def report(self):
        raise NotImplementedError;


class Exp_LSH(ExperientInterface):
    
    def run(self):
        self.comp = Composition(self.config)
        self.comp.load_5()
        self.residuals = self.comp.pm.T - np.dot(self.comp.wrm, self.comp.uvm.T)
        self.residuals = np.sqrt(np.sum(self.residuals*self.residuals,axis = 0))
        self.run_cosine()
        
    def run_cosine(self):
        com = np.dot(self.comp.wrm, self.comp.uvm.T)
        dp = np.sum(self.comp.pm.T * com,axis = 0)
        norm = np.sqrt(np.sum(com * com, axis = 0))
        self.residuals_cos = dp / norm
        
    def report_res_topk(self,k,path):
        res = [(self.residuals[i],i) for i in xrange(len(self.residuals))]
        res = sorted(res)
        f = open(path,'w')
        for i in xrange(min(k,len(res))):
            idx = res[i][1]
            r = res[i][0]
            phrase = self.comp.phrases[idx]
            f.write('{} {}\n'.format(r,phrase))
        f.close()

    def report_res_cos_dist(self):
        num_bins = 20
        n, bins, patches = plt.hist(self.residuals_cos, num_bins, normed = 1, facecolor = 'green')
        plt.show()
        

    def report_res_dist(self):
        # report the residuals distributions
        num_bins = 20
        n, bins, patches = plt.hist(self.residuals, num_bins, normed = 1, facecolor = 'green')
        plt.show()

    def report(self):
        self.report_res_dist()
        self.report_res_cos_dist()
        path = self.config.get('exp_lsh','path')
        k = self.config.getint('exp_lsh','topk')
        self.report_res_topk(k,path)


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_path = sys.argv[1]
    config = get_config(config_path)

    exp_lsh = Exp_LSH(config)
    exp_lsh.run()
    exp_lsh.report()

if __name__ == "__main__":
    main()

