from utils.config import get_config
import configparser
import sys

def fn2dict(path):
    d = {}
    f = open(path)
    for line in f:
        word = line.strip()
        d[word]=1
    return d

def main():
    config_path = sys.argv[1]
    config = get_config(config_path)
    
    noun_path = config.get('path','noun_words')
    adj_path = config.get('path','adj_words')
    bigram_path = config.get('path','bigram_words')
    stop_path = config.get('path','stop_words')
    new_path = config.get('path','bigram_adj_noun')
    fout = open(new_path,'w')

    dnoun = fn2dict(noun_path)
    dadj = fn2dict(adj_path)
    dstop = fn2dict(stop_path)
    
    with open(bigram_path) as f:
        for line in f:
            words = line.strip().split()
            if words[0] in dadj and words[1] in dnoun and (not words[0] in dstop) and (not words[1] in dstop):
                fout.write(line)

    fout.close()

if __name__ == '__main__':
    main()
