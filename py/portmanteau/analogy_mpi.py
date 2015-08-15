import sys
import numpy as np
from vector.LSH_sumbeam import LSH_sumbeam
from corpus.wordCollect import WordList
from utils.config import get_config
import logging
from nearpy.hashes import RandomBinaryProjections
from utils.heap import FixSizeHeap

from mpi4py import MPI
from portmanteau.analogy import build_environment,analogy

# Define MPI message tags
def enum(*sequential, **named):
    """
    Handy way to fake an enumerated type in Python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

tags = enum('READY', 'DONE', 'EXIT', 'START')

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

if rank == 0:
    # Master process executes code below
    config_fn = sys.argv[1]
    config = get_config(config_fn)

    port_path = config.get('portmanteau','path')
    ports = []
    for line in open(port_path):
        ll = line.split()
        w1 = ll[0]
        w2 = ll[1]
        ports.append((w1,w2))

    fout = open(config.get('portmanteau','outpath'),'w')

    num_tasks = len(ports)
    task_index = 0
    num_workers = size - 1
    closed_workers = 0
    print("Master starting with %d workers" % num_workers)
    while closed_workers < num_workers:
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if task_index < num_tasks:
                comm.send(ports[task_index], dest=source, tag=tags.START)
                print("Sending task %d to worker %d" % (task_index, source))
                task_index += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            w1,w2, w34s,norm = data
            if w34s != None:
                fout.write('\n####\n')
                fout.write('{} {} {}\n'.format(w1,w2,norm))
                for dis,w3,w4 in w34s:
                    fout.write('{} {} {}\n'.format(w3,w4,dis))
                fout.flush()
            print("Got data from worker %d" % source)
        elif tag == tags.EXIT:
            print("Worker %d exited." % source)
            closed_workers += 1

    fout.close()

    print("Master finishing")

else:

    # Worker processes execute code below
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    name = MPI.Get_processor_name()
    logging.info("I am a worker with rank %d on %s." % (rank, name))
    config_fn = sys.argv[1]
    config = get_config(config_fn)
    lsh,engine,matrix,wordlist = build_environment(config)
    naive = config.getint('portmanteau','naive')

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        
        if tag == tags.START:
            # Do the work here
            w1,w2 = data
            result,norm = analogy(w1,w2,lsh,engine,matrix,wordlist,naive=naive)
            comm.send((w1,w2,result,norm), dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)
    
