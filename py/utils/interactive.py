import numpy as np
from vector.bruteForce import BruteForceSearch
from utils.config import get_config
import vector.LSH
import logging
import exp.analysis

def cosd(v1,v2):
    d = np.dot(v1,v2)
    denom = np.linalg.norm(v1,2) * np.linalg.norm(v2,2)
    return d / denom


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config = get_config('../config/mac.cfg')
lsh = vector.LSH.LSH()
lsh.load_from_config(config)
ana = exp.analysis.Analysis()


