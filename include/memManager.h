#ifndef __MEM_MANAGER_H__
#define __MEM_MANAGER_H__
#include <stddef.h>
#define myDMalloc(fmt, var, size) fmt* var = (fmt*)ccmemMngr.borrowCache(size*sizeof(fmt))
#define myMalloc(fmt, var, size) var = (fmt*)ccmemMngr.borrowCache(size*sizeof(fmt))
#define myFree(ptr) ccmemMngr.returnCache(ptr); ptr = 0

class memManager{
  void* memoryp;
  void* storagep;
  void* maxstoragep;
  void* rentBookp;
  protected:
    virtual void c_malloc(void*&, size_t) = 0;
    virtual void c_memset(void*&, size_t) = 0;
  public:
    memManager();
    void* borrowCache(size_t);
    void* borrowCleanCache(size_t);
    void* borrowSame(void*);
    size_t getSize(void*);
    void* useOnsite(size_t); //no need to return, but you shouldn't ask for another borrow during the use of this pointer.
    void returnCache(void*);
    void registerMem(void*, size_t);
    void retrieveAll();
    ~memManager(){};
};

class ccMemManager : public memManager{
  void c_malloc(void* &ptr, size_t sz);
  void c_memset(void* &ptr, size_t sz);
  public:
    ccMemManager():memManager(){};
};

extern ccMemManager ccmemMngr;
#endif
