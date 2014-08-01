import heapq

class FixSizeHeap:
    
    def __init__(self,size):
        self.size = size
        self.data = []


    def push(self,item): # item should be (key,value)
        if len(self.data) >= self.size:
            if item > self.data[0]:
                heapq.heappop(self.data)
                heapq.heappush(self.data,item)
        else:
            heapq.heappush(self.data,item)

    def pop(self,item):
        return heapq.heappop(self.data)

def test():
    # keep track of largest 3 items
    heap = FixSizeHeap(3)
    for i in xrange(10):
        heap.push(i)
    print heap.data
    # keep track of smallest 3 items
    heap = FixSizeHeap(3)
    for i in xrange(10):
        heap.push((-i,i))
    print heap.data



if __name__=="__main__":
    test()
