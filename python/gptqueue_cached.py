#!/usr/bin/env python
import socket
import threading
from os.path import exists
import queue, sys, json
from mygpt4all import GPT4All, new_session, new_template
 
# 创建一个队列用于存放接收到的数据
data_queue = queue.Queue(maxsize=0)
 
# 定义TCP服务器的IP地址和端口号
HOST = 'localhost'
PORT = 8888

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
 
# 绑定服务器的IP地址和端口号
server_socket.bind((HOST, PORT))
 
# 开始监听客户端连接请求
server_socket.listen(1)

exitflag = 0

model = GPT4All('/home/boyang/.cache/gpt4all/mistral-7b-instruct-v0.1.Q4_0.gguf')
template = new_template()
 
# 创建一个TCP服务器套接字

def processData(handler):
    while not data_queue.empty():
        sock, user = data_queue.get()
        print(sock, user)
        words = ""
        with open(user+".request", "r") as qfile:
            words = qfile.read()
        if exists(user+".session"):
            with open(user+".session", "r") as sfile:
                session = json.load(sfile)
                response = model.generate(words, session, template)
        else:
            session = new_session()
            response = model.generate(words, session, template)
        with open(user+".session", "w") as sfile:
            json.dump(session, sfile, indent=4)
        with open(user+".response", "w") as rfile:
            rfile.write(response)
        print(session)
        sock.send("Done!".encode("utf-8"))
        sock.close()
        if user == "boyang" and words == "exit":
            exitflag = 1
    handler.working = 0

class datahandler:
    def __init__(self):
        self.working = 0
        self.t = 0
    def trigger(self):
        if self.working == 0:
            self.working = 1
            self.t = threading.Thread(target = processData, args=(self,))
            self.t.start()
 
def main():
    """主函数"""
    handler = datahandler()
    server_socket.settimeout(5)
    global exitflag
    while True:
        # 等待客户端连接请求
        # 创建一个线程处理客户端请求
        try:
            client_socket, address = server_socket.accept()
            """处理客户端请求的函数"""
            data = client_socket.recv(1024).decode('utf-8')
            data_queue.put([client_socket, data])
            handler.trigger()
        except socket.timeout:
            pass
        if exitflag:
            break
    # 关闭服务器套接字
    server_socket.close()
    sys.exit()
 
if __name__ == '__main__':
    main()
