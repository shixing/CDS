from flask import Flask
from flask.ext.restful import reqparse, abort, Api, Resource
from flask import request,make_response
import sys
import configparser
import urllib
import json
import hashlib
import time

from utils.config import get_config

import numpy as np
from vector.bruteForce import BruteForceSearch
from utils.config import get_config
import vector.LSH
import vector.LSH2gram
import logging
import exp.analysis
from nearpy.distances.angular import AngularDistance

# log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# get config
config_fn = sys.argv[1]
config = get_config(config_fn)

# lsh2gram
logging.info('H1')
lsh2gram = vector.LSH2gram.LSH2gram()
logging.info('H2')
lsh2gram.load_from_config_light(config)
logging.info('H3')

lsh2gram.engine_2gram.build_permute_index(200,10,500)

# app
logging.info('H4')
app = Flask(__name__)
api = Api(app)
logging.info('H5')


# decompose

class Decompose(Resource):
    def get(self,t):
        results = None
        print request.args
        print t
        if t == 'q1':
            qw = request.args['w']
            k = int(request.args['k'])
            naive = False
            if 'naive' in request.args:
                naive = True
            print qw,k,naive
            results = lsh2gram.query_1_2(qw,k,naive)
        if t == 'q2':
            qw1 = request.args['w1']
            qw2 = request.args['w2']
            k = int(request.args['k'])
            naive = False
            if 'naive' in request.args:
                naive = True
            results = lsh2gram.query_2_2(qw1,qw2,k,naive)
        return make_response(repr(results))

logging.info('H6')
api.add_resource(Decompose,'/api/decompose/<string:t>')

if __name__ == '__main__':
    logging.info('H7')
    app.run()
    logging.info('H8')
