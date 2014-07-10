# MySentences, the class to generate sentences one by one.

class MySentences:
    
    def __init__(self,filename):
        self.filename = filename
        
    def __iter__(self):
        with open(self.filename) as f:
            for line in f:
                ll = line.strip().lower().split()
                yield ll


