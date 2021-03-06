import numpy as np
from vector.bruteForce import BruteForceSearch
from utils.config import get_config
import vector.LSH
import vector.LSH2gram
import vector.LSH_sumbeam
import logging
import exp.analysis
from nearpy.distances.angular import AngularDistance

def cosd(v1,v2):
    d = np.dot(v1,v2)
    denom = np.linalg.norm(v1,2) * np.linalg.norm(v2,2)
    return d / denom

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
config = get_config('../config/mac.cfg')
lsh = vector.LSH_sumbeam.LSH_sumbeam()
lsh.load_from_config_light(config)
lsh.build_index_sumbeam(15)
lsh.build_permute_index(50,50,10)
lsh.query_1_2("reasoning",10,10)
#### ####
import numpy as np
from vector.engine2 import PermuteEngine
from nearpy.hashes import RandomBinaryProjections
from nearpy.hashes import RandomDiscretizedProjections
from nearpy.filters.nearestfilter import NearestFilter
from nearpy.distances.angular import AngularDistance

matrix = np.random.random((1000,200))
query = np.random.random(200)

dimension = 200
num_bit = 10
n = 1000


rbp = RandomBinaryProjections('rbp', num_bit)
# Create engine with pipeline configuration
engine = PermuteEngine(dimension, lshashes=[rbp])
engine.distance = AngularDistance()

for index in range(n):
    v = matrix[index]
    engine.store_vector(v, '%d' % index)

    
