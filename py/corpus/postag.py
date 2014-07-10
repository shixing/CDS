# Asynchronized Multi-machine version using MPICH
# parallel in two level, not three level.
# How to run:
# mpiexec -np 4 python -m corpus.postag ../../config/mac.cfg
#

import nltk
from mpi4py import MPI
from datetime import datetime
import sys
import os
import cPickle
import logging
from utils.config import get_config

def pos_tagging(file_in, file_text_out,file_pos_out,task_index,n):
    fin = open(file_in)
    fposout = open(file_pos_out,'w')
    ftextout = open(file_text_out,'w')
    i = 0
    for line in fin:
        if i%n != task_index:
            i+=1
            continue
        line = line.lower()
        sts = nltk.sent_tokenize(line)
        for st in sts:
            text = nltk.word_tokenize(st)
            pos = nltk.pos_tag(text)
            pos_string = ' '.join([x[1] for x in pos])
            text_string = ' '.join([x[0] for x in pos])
            fposout.write( pos_string + ' ' )
            ftextout.write( text_string + ' ')
        fposout.write('\n')
        ftextout.write('\n')
        i += 1
        if i % 10000 == 0:
            logging.info('Tagging #{}'.format(i))

    fin.close()
    fposout.close()
    ftextout.close()

def combine_file(path_prefix,n,output_file):
    fins = [open(path_prefix+'.'+str(x)) for x in xrange(n)]
    fout = open(output_file,'w')
    finished_file = 0
    finished_files = {}
    while finished_file < n:
        for i,fin in enumerate(fins):
            if i in finished_files:
                continue
            line = fin.readline()
            if not line:
                finished_file += 1
                finished_files[i] = 1
            else:
                fout.write(line)
    fout.close()
    

def enum(*sequential, **named):
    """
    Handy way to fake an enumerated type in Python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)



def main():
    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    tags = enum('READY', 'DONE', 'EXIT', 'START')

    if rank == 0:
        # Master process
        '''
        $1 path to config file
        '''
        start = datetime.now()
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # loading configs;
        config = get_config(sys.argv[1])
        nthread = config.getint('pos','nthread')
        #datafile = config.get('path','short_abstracts')
        datafile = config.get('test','test_data')
        nworkers = nthread - 1

        closed_workers = 0
        task_index = 0
        
        while closed_workers < size-1:
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == tags.READY:
                # Worker is ready, so send it a task
                if task_index < nworkers:
                    input_file = datafile
                    output_file = '{}.pos.{}'.format(datafile,task_index)
                    text_output_file = '{}.text.{}'.format(datafile,task_index)
                    comm.send((input_file,text_output_file,output_file,task_index,nworkers), dest=source, tag=tags.START)
                    logging.info("Sending task %d to worker %d" % (task_index, source))
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)
            elif tag == tags.DONE:
                logging.info("Got data from worker %d" % source)
            elif tag == tags.EXIT:
                logging.info("Worker %d exited." % source)
                closed_workers += 1
        

        logging.info("Combine the pos files")
        combine_file(datafile+'.pos',nworkers,datafile+'.pos.combine')
        logging.info("Combine the text files")
        combine_file(datafile+'.text',nworkers,datafile+'.text.combine')
        logging.info("Master finishing")
        
        
        end = datetime.now()
        print end-start

    else:
        # Worker process
        rank = comm.rank
        name = MPI.Get_processor_name()
        logging.info("I am a worker with rank %d on %s." % (rank, name))
        while True:
            comm.send(None, dest=0, tag=tags.READY)
            data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            
            if tag == tags.START:
                # Do the work here
                input_file,text_output_file,output_file, index, n = data
                pos_tagging(input_file,text_output_file,output_file,index,n)
                comm.send(None, dest=0, tag=tags.DONE)
            elif tag == tags.EXIT:
                break
            
        comm.send(None, dest=0, tag=tags.EXIT)
            


    


if __name__ == '__main__':
    main()
