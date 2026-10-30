#include <stdio.h>
#include "memManager.hpp"
#include <stdlib.h>
#include <string.h>
#include "fmt/core.h"

ccMemManager ccmemMngr;

void* memManager::borrowCache(size_t sz){
  void* ret;
  auto &nele = count[sz];
  auto &vec = pool[sz];
  if(nele != 0) {
    ret = vec[--nele];
  } else {
    c_malloc(ret, sz);
  }
  rentBook[ret] = sz;
  return ret;
}

void* memManager::borrowCleanCache(size_t sz){
  void* ret = borrowCache(sz);
  c_memset(ret, sz);
  return ret;
}

void memManager::registerMem(void* ptr, size_t sz){
  count[sz];  // ensure key exists
  pool[sz];
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
    fmt::println("Pointer {} not found in rentBook (not managed or already returned).", mem);
    abort();
  }
  return borrowCache(iter->second);
}

size_t memManager::getSize(void* mem){
  auto iter = rentBook.find(mem);
  if(iter == rentBook.end()) {
    fmt::println("Pointer {} not found in rentBook (not managed or already returned).", mem);
    abort();
  }
  return iter->second;
}

void memManager::returnCache(void* mem){
  if(!mem) return;
  auto iter = rentBook.find(mem);
  if(iter == rentBook.end()) {
    fmt::println("Pointer {} not found in rentBook (not managed or already returned).", mem);
    fmt::println("rentBook:");
    for(auto& [ptr, sz] : rentBook)
      fmt::println("  {} -> {}", ptr, sz);
    abort();
  }
  size_t sz = iter->second;
  rentBook.erase(iter);
  auto &nele = count[sz];
  auto &maxele = maxcount[sz];
  auto &vec = pool[sz];
  if(nele == maxele) {
    maxele++;
    vec.push_back(mem);
  } else {
    vec[nele] = mem;
  }
  nele++;
}

void ccMemManager::c_malloc(void* &ptr, size_t sz){ ptr = malloc(sz); }
void ccMemManager::c_memset(void* &ptr, size_t sz){ memset(ptr, 0, sz); }
