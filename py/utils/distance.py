from bitarray import bitarray
import numpy as np

def hamming_distance(a,b):
    return int((a ^ b).count())

def cosine_vv(a,b):
    # a,b belongs to np.array([1,2,3])
    # return float
    dt = np.dot(a,b)
    norma = np.sqrt(sum(a*a))
    normb = np.sqrt(sum(b*b))
    return dt/norma/normb
    
def cosine_vm(self,vec,matrix):
    # a belongs to np.array([1,2])
    # b belongs to np.array([[1,2],[-1,-2]])
    # return np.array([1,-1])
    
    dt = np.dot(matrix,vec.T)
    denom = np.linalg.norm(vec,2)*np.sqrt(np.sum(matrix * matrix, axis = 1))
    dists = dt / denom
    return dists


def cosine_distance(a,b):
    # a,b should be matrix
    # each row is a vector in a, b
    dt = np.dot(a,b.T)
    norm_a = np.sqrt(np.sum(a * a, axis = 1))
    norm_a = norm_a.reshape((len(norm_a),1))
    norm_b = np.sqrt(np.sum(b * b, axis = 1))
    norm_b = norm_b.reshape((len(norm_b),1))
    cos_matrix = dt / ( np.dot( norm_a , norm_b.T))
    return cos_matrix

