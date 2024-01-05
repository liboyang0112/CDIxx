#!/usr/bin/env python
import socket
from sys import argv
sk = socket.socket()
sk.connect(('localhost',8888))
#send port number
info = argv[1]
print(info)
sk.send(info.encode('utf-8'))
print("message sent")
ret = sk.recv(1024).decode('utf-8')
print(ret)
sk.close()
