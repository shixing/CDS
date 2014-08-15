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
from 
# get config
config_fn = sys.argv[1]
config = get_config(config_fn)

# app
app = Flask(__name__)
api = Api(app)

# LSH2gram

# decompose

class Decompose(Resource):
    def get(self):
        eword = request.args['t']
        
