#ifndef __MEM_MANAGER_H__
#define __MEM_MANAGER_H__
#include <stddef.h>
#include <unordered_map>
#include <vector>

#define myDMalloc(fmt, var, size) fmt* var = (fmt*)ccmemMngr.borrowCache((size)*sizeof(fmt))
#define myMalloc(fmt, var, size) var = (fmt*)ccmemMngr.borrowCache((size)*sizeof(fmt))
#define myMallocClean(fmt, var, size) var = (fmt*)ccmemMngr.borrowCleanCache((size)*sizeof(fmt));
#define myFree(ptr) ccmemMngr.returnCache(ptr); ptr = 0

class memManager{
  std::unordered_map<size_t, std::vector<void*>> pool;
  std::unordered_map<size_t, size_t> count;      // available blocks per size
  std::unordered_map<size_t, size_t> maxcount;   // total capacity per size
  std::unordered_map<void*, size_t> rentBook;    // live pointer -> size
  protected:
    virtual void c_malloc(void*&, size_t) = 0;
    virtual void c_memset(void*&, size_t) = 0;
  public:
    void* borrowCache(size_t);
    void* borrowCleanCache(size_t);
    void* borrowSame(void*);
    size_t getSize(void*);
    void* useOnsite(size_t);
    void returnCache(void*);
    void registerMem(void*, size_t);
    void retrieveAll();
    ~memManager() = default;
};

class ccMemManager : public memManager{
  void c_malloc(void* &ptr, size_t sz) override;
  void c_memset(void* &ptr, size_t sz) override;
  public:
    ccMemManager() = default;
};

extern ccMemManager ccmemMngr;
#endif
