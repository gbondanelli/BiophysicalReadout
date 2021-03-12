from numpy import *
from numpy import random
import multiprocessing as mp

def collect_result(result):
    global results
    results.append(result)

def function2(j,N):
    s = 0
    for i in range(N):
        a = random.normal(0,1/sqrt(N),(N,N))
        b = random.normal(0,1/sqrt(N),(N,N))
        s += sum(dot(a,b))
    return s

def function_to_par(n,N):
    pool = mp.Pool(mp.cpu_count())
    a = []
    # for i in range(n):
    #     s = function2(N)
    #     a.append(s)
    for i in range(n):
        pool.apply_async(function2, args = (i,N), callback = collect_result)
    pool.close()
    return a


##

