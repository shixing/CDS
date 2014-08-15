from bitstring import BitArray

class OrderableBitArray(BitArray):
    def __init__(self,string):
        super(BitArray,self).__init__('0b'+string)
    
    def __lt__(self,other):
        return self.bin < other.bin
    
    def __gt__(self,other):
        return self.bin > other.bin

    def __ge__(self,other):
        return self.bin >= other.bin
    
    def __le__(self,other):
        return self.bin <= other.bin

    def hamming_distance(self,other):
        assert(len(self) == len(other))
        d = 0
        for i in xrange(len(self)):
            if self[i] != other[i]:
                d += 1
        return d
    
    def __hash__(self):
        return self.bin.__hash__()
