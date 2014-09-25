import sys
import numpy as np
from vector.LSH_sumbeam import LSH_sumbeam
from corpus.wordCollect import WordList
from utils.config import get_config
import logging
from nearpy.hashes import RandomBinaryProjections
from utils.heap import FixSizeHeap



def build_environment(config):
    lsh = LSH_sumbeam()
    lsh._load_external_variable(config,composition=False)
    lsh._filter_wordlist(config)

    # combine top 20k noun and 20k adj into a single wordlist
    wordlist = WordList()
    wordlist.words = lsh.wordlist1.words + lsh.wordlist2.words
    wordlist.build_index()

    # build a matrix
    matrix = lsh._list2matrix_w2v(wordlist,lsh.w2v)

    # build an engine
    dim = np.shape(matrix)[1]
    num_bits = 15
    rbp = RandomBinaryProjections('rbp', num_bits)
    rbp.reset(dim)    
    engine = lsh._build_rbp_permute_engine(matrix,rbp)
    num_permutation = 50
    beam_size = 50
    num_neighbour = 100
    engine.build_permute_index(num_permutation,beam_size,num_neighbour)
    
    return lsh,engine,matrix,wordlist

def vector_norm(v):
    v = np.array(v)
    return np.sqrt(np.dot(v,v))

def analogy(w1,w2,lsh,engine,matrix,wordlist,naive=False):
    if naive:
        return analogy_naive(w1,w2,lsh,engine,matrix,wordlist)
    else:
        return analogy_lsh(w1,w2,lsh,engine,matrix,wordlist)

def analogy_naive(w1,w2,lsh,engine,matrix,wordlist):
    n = engine.hamming_beam_size
    vector1 = lsh.w2v.getNorm(w1)
    vector2 = lsh.w2v.getNorm(w2)
    
    if vector1 == None or vector2 == None:
        return None
    delta = vector2-vector1
    delta_norm = vector_norm(delta)
    print vector_norm(vector1)
    print vector_norm(vector2)
    print delta_norm
    heap = FixSizeHeap(1000)
        
    j = 0
    for w3 in wordlist.words:
        j += 1
        if j % 1000 == 0:
            logging.info('Searching: {}/{}'.format(j,len(wordlist.words)))

        idx = wordlist.index[w3]
        vector3 = matrix[idx,:]
        vector4 = vector3 + delta
        topn, dists = lsh.query1_naive_matrix(vector4,matrix)
        
        
        for i in xrange(n):
            idx = topn[i]
            dis = 1-dists[idx]
            w4 = wordlist.words[idx]
            if w3 == 'previous':
                print w4,dis
            if w4 == w2 or w3 == w4:
                continue
            else:
                heap.push((-dis,w3,w4))
    data = list(set(heap.data))
    data = sorted(data,key = lambda x:-x[0] )
    return data,delta_norm


def analogy_lsh(w1,w2,lsh,engine,matrix,wordlist):
    vector1 = lsh.w2v.getNorm(w1)
    vector2 = lsh.w2v.getNorm(w2)
    
    if vector1 == None or vector2 == None:
        return None
    delta = vector2-vector1
    delta_norm = vector_norm(delta)
    print vector_norm(vector1)
    print vector_norm(vector2)
    print delta_norm
    heap = FixSizeHeap(1000)
        
    j = 0
    for w3 in wordlist.words:
        j += 1
        if j % 1000 == 0:
            logging.info('Searching: {}/{}'.format(j,len(wordlist.words)))

        idx = wordlist.index[w3]
        vector3 = matrix[idx,:]
        vector4 = vector3 + delta
        candidates = engine.neighbours2(vector4)
        for vec,idx,dis in candidates:
            idx = int(idx)
            w4 = wordlist.words[idx]
            if w4 == w2 or w3 == w4:
                continue
            else:
                heap.push((-dis,w3,w4))
    data = list(set(heap.data))
    data = sorted(data,key = lambda x:-x[0] )
    return data,delta_norm


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    config_fn = sys.argv[1]
    config = get_config(config_fn)
    
    lsh,engine,matrix,wordlist = build_environment(config)

    # load the portmantaeu word list
    port_path = config.get('portmanteau','path')
    ports = []
    for line in open(port_path):
        ll = line.split()
        w1 = ll[0]
        w2 = ll[1]
        ports.append((w1,w2))

    logging.info('Loaded {} portmanteaus'.format(len(ports)))
        
    # search according to ports
    fout = open(config.get('portmanteau','outpath'),'w')
    i = 0
    for w1,w2 in ports:
        i += 1
        logging.info('search for {}/{}:({} {})'.format(i,len(ports),w1,w2))
        data,norm = analogy(w1,w2,lsh,engine,matrix,wordlist)
        if data == None:
            continue
        fout.write('\n####\n')
        fout.write('{} {} {}\n'.format(w1,w2,norm))
        for dis,w3,w4 in data:
            fout.write('{} {} {}\n'.format(w3,w4,-dis))
        if i == 2:
            break

    fout.close()
                       
        


    

if __name__ == '__main__':
    main()
    
