import random
from bitarray import bitarray
from kmeans.kmeans_basic import kmeans

def average_distance(kmh):
    s = sum([kmh[2][i][kmh[1][i]] for i in xrange(len(kmh[1]))])
    s = s * 1.0 / len(kmh[2])
    return s

def get_different_random_numbers(n,m):
    '''
    n = [0,1,...,n-1]
    return m different random numbers from n
    '''
    l = range(n)
    for end in xrange(n - 1, n-m-1, -1):
        r = random.randint(0, end)
        tmp = l[end]
        l[end] = l[r]
        l[r] = tmp
    return l[n-m:]

def random_bitarray(n):
    b = bitarray()
    for i in xrange(n):
        b.append(random.choice([True,False]))
    return b

def hamming_dist(a,b):
    return int((a ^ b).count())

def hamming_centroid_f(data):
    min_d = 2 * len(data[0])
    centroid = data[0]
    for c in data:
        sum_d = 0
        for d in data:
            sum_d += hamming_dist(c,d)
        if sum_d < min_d :
            min_d = sum_d
            centroid = c
    return c

def kmeans_hamming_restart(numbers,k,num_restart):    
    # numbers are list of bitarrays
    j = 0
    m = len(numbers[0])
    n = len(numbers)
    avd_min = m*n/k
    kmh_final = None
    while j<20:
        rand_index = get_different_random_numbers(n,k)
        cents = [numbers[x] for x in rand_index]
        kmh = kmeans(numbers,cents,hamming_dist,hamming_centroid_f,0)
        avd = average_distance(kmh)
        print j,avd,avd_min
        if avd < avd_min:
            avd_min = avd
            kmh_final = kmh
        j += 1
    return kmh_final

def test():
    m = 30
    n = 1000
    k = 100
    numbers = []
    for i in xrange(n):
        r = random_bitarray(m)
        numbers.append(r)
    kmeans_hamming_restart(numbers,k,20)

if __name__ == '__main__':
    test()
