import numpy as np

def objective1(vec,matrix):
    # cos(w4,w2+w3-w1)
    dt = np.dot(matrix,vec.T)
    denom = np.linalg.norm(vec,2)*np.sqrt(np.sum(matrix * matrix, axis = 1))
    dists = dt / denom
    topn = np.argsort(dists)[::-1]
    return topn,dists

def objective2(w1,w2,w3,matrix):
    #cos(w4,w3) + cos(w4,w2) + cos(w3,w1) - cos(w4,w1) - cos(w3,w2)
    dists = np.dot(matrix,w3+w2-w1) + np.dot(w3,w1-w2)
    topn = np.argsort(dists)[::-1]
    return topn,dists
        
def objective3(w1,w2,w3,matrix):
    # log(cos(w4,w3)) + log(cos(w4,w2)) + log(cos(w3,w1)) - log(cos(w4,w1)) - log(cos(w3,w2))
    dists = np.log(1+np.dot(matrix,w3)) + \
        np.log(1+np.dot(matrix,w2)) + \
        np.log(1+np.dot(w3,w1)) + \
        - np.log(1+np.dot(matrix,w1)) + \
        - np.log(1+np.dot(w3,w2))
    topn = np.argsort(dists)[::-1]
    return topn,dists
    
                 
