#include <cstdio>
#include "memManager.h"
#include <map>
#include <vector>
#include <cstdlib>
#include <string.h>

ccMemManager ccmemMngr;

#define memory (*(std::map<size_t, std::vector<void*>>*) memoryp)
#define storage (*(std::map<size_t, int>*) storagep)
#define maxstorage (*(std::map<size_t, int>*) maxstoragep)
#define rentBook (*(std::map<void*, size_t>*) rentBookp)
memManager::memManager(){
  memoryp = new std::map<size_t, std::vector<void*>>();
  storagep = new std::map<size_t, int>();
  maxstoragep = new std::map<size_t, int>();
  rentBookp = new std::map<void*, size_t>();
}

void* memManager::borrowCache(size_t sz){
  void *ret;
  auto iter = storage.find(sz);
  if(iter == storage.end()) {
    iter = storage.emplace(sz,0).first;
    maxstorage.emplace(sz,0);
    memory.emplace(sz, std::vector<void*>());
  }
  if(iter->second!=0) {
    ret = memory[sz][--iter->second];
  }else{
    c_malloc(ret, sz);
  }
  void* bb = ret;
  rentBook[bb] = sz;
  return ret;
}

void* memManager::borrowCleanCache(size_t sz){
  void* ret = borrowCache(sz);
  c_memset(ret, sz);
  return ret;
}

void memManager::registerMem(void* ptr, size_t sz){
  auto iter = storage.find(sz);
  if(iter == storage.end()) {
    iter = storage.emplace(sz,0).first;
    maxstorage.emplace(sz,0);
    memory.emplace(sz, std::vector<void*>());
  }
  rentBook[ptr] = sz;
}

void* memManager::useOnsite(size_t sz){
  void* ret = borrowCache(sz);
  returnCache(ret);
  return ret;
}

void* memManager::borrowSame(void* mem){
  auto iter = rentBook.find(mem);
  if(iter == rentBook.end()) {
    printf("This pointer %p, is not found in the rentBook, please check if the memory is managed by memManager or returned already.\n", mem);
    abort();
  }
  int siz = iter->second;
  return borrowCache(siz);
}

size_t memManager::getSize(void* mem){
  auto iter = rentBook.find(mem);
  if(iter == rentBook.end()) {
    printf("This pointer %p, is not found in the rentBook, please check if the memory is managed by memManager or returned already.\n", mem);
    abort();
  }
  return iter->second;
}

void memManager::returnCache(void* mem){
  if(mem == NULL) return;
  auto iter = rentBook.find(mem);
  if(iter == rentBook.end()) {
    printf("This pointer %p, is not found in the rentBook, please check if the memory is managed by memManager or returned already.\n", mem);
    printf("rent book content:\n");
    for(auto x : rentBook){
      printf("%p, %ld\n", x.first, x.second);
    }
    abort();
  }
  int siz = iter->second;
  auto &nele = storage[siz];
  auto &maxele = maxstorage[siz];
  if(nele == maxele) {
    maxele++;
    memory[siz].push_back(mem);
  }else{
    memory[siz][nele] = mem;
  }
  nele++;
  rentBook.erase(iter);
}
void ccMemManager::c_malloc(void* &ptr, size_t sz){ptr = malloc(sz);}
void ccMemManager::c_memset(void* &ptr, size_t sz){memset(ptr, 0, sz);}
