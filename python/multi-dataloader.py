#!/bin/env python
import time
import multiprocessing
from cythonLoader import cythonLoader as clder

def task(loader, i, send_end):
    send_end.send(loader.read(i))

if __name__ == '__main__':

    start = time.perf_counter()
    loader = clder("traindb")
    print(loader.read(9))
    recv_end1, send_end1 = multiprocessing.Pipe(False)
    p1 = multiprocessing.Process(target=task,args=(loader, 10, send_end1))
    recv_end2, send_end2 = multiprocessing.Pipe(False)
    p2 = multiprocessing.Process(target=task,args=(loader, 11, send_end2))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(recv_end1.recv())
    print(recv_end2.recv())

    finish = time.perf_counter()
    print(f'It took {finish-start:.2f} second(s) to finish')
