import sys,os
import gensim, logging
import configparser
from utils.config import *
import numpy as np
from word2vec import MyWord2Vec
import datetime

class Composition:
    
    def __init__(self,config):
        self.config = config
    

    def load_data(self,fnDict,N,model):
        self.N = N
        logging.info("Constructing P and UV")
        f = open(fnDict)
        i = 0
        pm = None
        uvm = None
        for line in f:
            phrase_count = line.split(' ')
            ll = phrase_count[0].split('_')
            if len(ll)!=2:
                continue
            pw = phrase_count[0].strip()
            uw = ll[0]
            vw = ll[1]
            p,u,v = None,None,None;
            try:
                p = model[pw]
                u = model[uw]
                v = model[vw]
            except Exception as e:
                #print e
                logging.info("{} {} {}".format(pw,uw,vw))
                continue
            uv = np.hstack([u,v])
            if pm == None:
                pm = p
            else:
                pm = np.vstack([pm,p])
            if uvm == None:
                uvm = uv
            else:
                uvm = np.vstack([uvm,uv])
            i += 1
            if i % 10000 == 0:
                logging.info("Processing {} phrases".format(i))
            if i >= N:
                break

        self.pm = pm
        self.uvm = uvm
        

    def train(self):
        # use self.uvm self.pm to calculate self.wrm
        # wrm = matrix of W_R
        # rm = matrix of residuals
        logging.info("Training Composition Matrix W_R")
        wrm, rm, _ , _= np.linalg.lstsq(self.uvm,self.pm)
        self.wrm = wrm.T
        self.rm = rm.T

        self.residuals = np.linalg.norm(self.pm.T - np.dot(self.wrm,self.uvm.T),2) / self.N
        logging.info("Average Residuals = {}".format(self.residuals))

    def save(self,model_path):
        # save the model
        np.save(model_path+'.wrm.npy',self.wrm)
        logging.info("Saving W_R({})".format(model_path+'.wrm.npy'))
        np.save(model_path+'.rm.npy',self.rm)
        logging.info("Saving Residuals({})".format(model_path+'.rm.npy'))
        np.save(model_path+'.uvm.npy',self.uvm)
        logging.info("Saving UV({})".format(model_path+'.uvm.npy'))
        np.save(model_path+'.pm.npy',self.pm)
        logging.info("Saving P({})".format(model_path+'.pm.npy'))


    def load(self,model_path):
        logging.info("Loading models")
        self.wrm = np.load(model_path+'.wrm.npy')
        self.rm = np.load(model_path+'.rm.npy')
        self.uvm = np.load(model_path+'.uvm.npy')
        self.pm = np.load(model_path+'.pm.npy')




    def train_process(self):
        start = datetime.datetime.now()
        fnDict = self.config.get('path','short_abstracts_text') + '.phrase.dict'
        model_path = self.config.get('word2vec','model_path')
        N = self.config.getint('composition','N')
        model_composition_path = self.config.get('composition','model_path')
        
        myWord2Vec = MyWord2Vec()
        myWord2Vec.load(self.config)
        
        self.load_data(fnDict,N,myWord2Vec.model)
        self.train()
        self.save(model_composition_path)
        
        end = datetime.datetime.now()        
        logging.info("{}".format(start-end))


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_path = sys.argv[1]
    config = get_config(config_path)
    
    composition = Composition(config)
    composition.train_process()

if __name__ == "__main__":
    main()
