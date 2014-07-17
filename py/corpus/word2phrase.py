# 1 POS tags
# 2 Scan A + N, find top 50K phrases
# replace top 50K phrases

import sys,os
import nltk
import cPickle
from utils.config import get_config
import configparser
import logging

def pos_tagging(file_in, file_out):
    fin = open(file_in)
    fout = open(file_out,'w')
    i = 0
    for line in fin:
        sts = nltk.sent_tokenize(line)
        for st in sts:
            text = st.split()
            pos = nltk.pos_tag(text)
            pos_string = ' '.join([x[1] for x in pos])
            fout.write( pos_string + ' ' )
        fout.write('\n')
        i += 1
        if i % 1000 == 0:
            logging.info('Tagging #{}'.format(i))

    fin.close()
    fout.close()

def top_AN(text_file,pos_file,dict_file):    
    jj = set(['JJ','JJS','JJR'])
    nn = set(['NN','NNS'])
    phrase_dict = {}

    fpos = open(pos_file)

    k = 0
    for line in open(text_file):
        pos_string = fpos.readline()
        pos = pos_string.strip().split()
        text = line.strip().split()
        if (len(text)!=len(pos)):
            print line
            print pos
            print len(text), len(pos)
            print k
            assert(len(text)==len(pos))
            
            
        for i in xrange(len(pos)-1):
            p0 = pos[i]
            p1 = pos[i+1]
            if p0 in jj and p1 in nn:
                phrase = (text[i],text[i+1])
                if not phrase in phrase_dict:
                    phrase_dict[phrase] = 0
                phrase_dict[phrase] += 1
        k += 1
        if k % 10000 == 0:
            logging.info('Collecting phrases #{}'.format(k))

    fpos.close()    

    # sorting
    logging.info('Sorting {} phrases'.format(len(phrase_dict)))
    phrases= []
    for key in phrase_dict:
        count = phrase_dict[key]
        phrases.append((count,key))
    phrases = sorted(phrases,reverse=True)

    # saving
    logging.info('Saving {} phrases'.format(len(phrase_dict)))
    fout = open(dict_file,'w')
    for phrase in phrases:
        fout.write(phrase[1][0]+'_'+phrase[1][1]+' '+str(phrase[0])+'\n')
        
    fout.close()
    

def load_dict(dict_file,topn):
    phrase_dict = {}
    i = 0
    for line in open(dict_file):
        ll = line.strip().split()
        count = int(ll[1])
        phrase_dict[ll[0]] = count
        i += 1
        if i>= topn:
            break

    return phrase_dict

def replace_phrase(file_word,file_phrase,file_dict,topn):
    fword = open(file_word)
    fphrase = open(file_phrase,'w')
    phrase_dict = load_dict(file_dict,topn)
    k = 0
    for line in fword:
        text = line.strip().split()
        i = 0
        temp = []
        replaced = False
        while i<len(text):
            if i == len(text) - 1:
                temp.append(text[i])
                break
            phrase = text[i] + '_' + text[i+1]
            if phrase in phrase_dict:
                temp.append(phrase)
                replaced = True
                i += 2
            else:
                temp.append(text[i])
                i += 1
        if replaced:
            fphrase.write(' '.join(temp)+'\n')
            fphrase.write(line)
        else:
            fphrase.write(' '.join(temp)+'\n')
        
        k += 1
        if k % 10000 == 0:
            logging.info('Replacing #{}'.format(k))


    fphrase.close()

def test_pos_tagging():    
    ftext = '/Users/xingshi/Workspace/misc/CDS/data/100.text.combine'
    fpos = '/Users/xingshi/Workspace/misc/CDS/data/100.pos.combine'
    fdict = '/Users/xingshi/Workspace/misc/CDS/data/100.phrase.dict'
    fphrase = '/Users/xingshi/Workspace/misc/CDS/data/100.phrase'
    #pos_tagging(ftext,fpos)
    top_AN(ftext,fpos,fdict)
    replace_phrase(ftext,fphrase,fdict,50000)

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    config_fn = sys.argv[1]
    config = get_config(config_fn)
    ftext = config.get('path','short_abstracts_text')
    fpos = config.get('path','short_abstracts_pos')
    fdict = ftext + '.phrase.dict'
    fphrase = ftext + '.phrase'
    #logging.info('POS tagging...')
    #pos_tagging(ftext,fpos)
    logging.info('collecting phrases...')
    top_AN(ftext,fpos,fdict)
    logging.info('replacing phrases...')
    replace_phrase(ftext,fphrase,fdict,50000)



if __name__ == '__main__':
#    test_pos_tagging()
    main()
