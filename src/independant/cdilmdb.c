#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <memory.h>
#include <lmdb.h>
#include <zlib.h>

#define E(expr) CHECK((rc = (expr)) == MDB_SUCCESS, #expr)
#define RES(err, expr) ((rc = expr) == (err) || (CHECK(!rc, #expr), 0))
#define CHECK(test, msg) ((test) ? (void)0 : ((void)fprintf(stderr, \
        "%s:%d: %s: %s\n", __FILE__, __LINE__, msg, mdb_strerror(rc)), abort()))

//data structure: ndata, data_size[0], data_size[1], ..., data[0], data[1], ...
int rc;
struct lmdbds{
  MDB_env *env;
  MDB_dbi dbi;
  MDB_val key, data_LMDB;
  MDB_txn *txn;
  MDB_stat mst;
  size_t data_cache_size;
  size_t mvdata_size;
  void* data_cache;
  char compress;
};
struct lmdbds* dsList[100];
void setCompress(int handle){
  struct lmdbds* ds = dsList[handle];
  ds->compress = 1;
}
int unoccupied = 0;
int initLMDB(int* handle, const char* dbDIR){
  mkdir(dbDIR, S_IRWXU | S_IRWXG | S_IRWXO);
  struct lmdbds *ds = (struct lmdbds*) malloc(sizeof(struct lmdbds));
  ds -> data_cache_size = 0;
  ds -> mvdata_size = 0;
  ds -> data_cache = 0;
  ds -> compress = 0;
  *handle = unoccupied;
  dsList[unoccupied] = ds;
  unoccupied+=1;
  E(mdb_env_create(&ds->env));
  E(mdb_env_set_maxreaders(ds->env, 1));
  E(mdb_env_set_mapsize(ds->env, 10485760000));
  E(mdb_env_open(ds->env, dbDIR, MDB_WRITEMAP/*|MDB_NOSYNC*/, 0664));
  E(mdb_env_stat(ds->env, &ds->mst));
  E(mdb_txn_begin(ds->env, NULL, /*MDB_WRITEMAP*/0, &ds->txn));
  E(mdb_dbi_open(ds->txn, NULL, 0, &ds->dbi));
  return ds->mst.ms_entries;
}
int fillLMDBCoded(int handle, int *keyval, void* data, size_t data_size)
{
  struct lmdbds* ds = dsList[handle];
  ds->key.mv_size = sizeof(int);
  ds->key.mv_data = keyval;
  if(ds->mvdata_size < data_size){
    ds->mvdata_size = data_size;
    if(ds->data_LMDB.mv_data) free(ds->data_LMDB.mv_data);
    ds->data_LMDB.mv_data = malloc(data_size);
  }
  ds->data_LMDB.mv_size = ds->mvdata_size;
  int ec = compress(ds->data_LMDB.mv_data, &ds->data_LMDB.mv_size, data, data_size);
  if(ec < Z_OK){
    fprintf(stderr,"compress failed with ec %d\n", ec);
    exit(0);
  }
  ec =  mdb_put(ds->txn, ds->dbi, &ds->key, &ds->data_LMDB,0);
  if(ec!=MDB_SUCCESS){
    fprintf(stderr,"mdb_put failed with ec %d\n", ec);
    exit(0);
  }
  return ec;
}
int fillLMDB(int handle, int *keyval, int ndata, void** data, size_t* data_size)
{
  struct lmdbds* ds = dsList[handle];
  size_t totalSize = sizeof(int)+ndata*sizeof(size_t);
  for(int i = 0; i < ndata; i++)
    totalSize+=data_size[i];
  if(totalSize > ds->data_cache_size){
    ds->data_cache_size = totalSize;
    free(ds->data_cache);
    ds->data_cache = malloc(ds->data_cache_size);
  }
  memcpy(ds->data_cache, &ndata, sizeof(int));
  size_t shift=sizeof(int);
  memcpy(ds->data_cache+shift, data_size, ndata*sizeof(size_t));
  shift+=ndata*sizeof(size_t);
  for(int i = 0; i < ndata; i++){
    memcpy(ds->data_cache+shift, data[i], data_size[i]);
    shift+=data_size[i];
  }
  if(!ds->compress){
    ds->key.mv_size = sizeof(int);
    ds->key.mv_data = keyval;
    ds->data_LMDB.mv_data = ds->data_cache;
    ds->data_LMDB.mv_size = totalSize;
    int ec =  mdb_put(ds->txn, ds->dbi, &ds->key, &ds->data_LMDB,0);
    if(ec!=MDB_SUCCESS){
      fprintf(stderr,"mdb_put failed with ec %d\n", ec);
      exit(0);
    }
    return ec;
  }
  else return fillLMDBCoded(handle, keyval, ds->data_cache, totalSize);
}

void saveLMDB(int handle){
  struct lmdbds* ds = dsList[handle];
  E(mdb_txn_commit(ds->txn));
  mdb_dbi_close(ds->env, ds->dbi);
  E(mdb_txn_begin(ds->env, NULL, MDB_RDONLY, &ds->txn));
  E(mdb_dbi_open(ds->txn, NULL, 0, &ds->dbi));
}
void readLMDB(int handle, int *ndata, void*** data, size_t** data_size, int *keyval){
  struct lmdbds* ds = dsList[handle];
  if(keyval){
    ds->key.mv_size = sizeof(int);
    ds->key.mv_data = keyval;
  }
  if((rc = mdb_get(ds->txn, ds->dbi , &ds->key, &ds->data_LMDB)) != MDB_SUCCESS){
    if(rc==MDB_NOTFOUND){
      fprintf(stderr,"data not found with ds->key %d\n",*keyval);
    }
    fprintf(stderr,"data read failed with EC %d\n",rc);
    exit(0);
  }
  int err;
  void* dataptr = ds->data_LMDB.mv_data;
  if(ds->compress){
    if(ds->data_cache_size == 0) {
      ds->data_cache_size = ds->data_LMDB.mv_size;
      ds->data_cache = malloc(ds->data_cache_size);
    }
    while((err = uncompress(ds->data_cache, &ds->data_cache_size, ds->data_LMDB.mv_data, ds->data_LMDB.mv_size)) != Z_OK){
      if(err != Z_BUF_ERROR) {
        fprintf(stderr,"zip error: %d\n", err);
        exit(0);
      }
      ds->data_cache_size *= 1.2;
      free(ds->data_cache);
      ds->data_cache = malloc(ds->data_cache_size);
    }
    dataptr = ds->data_cache;
  }
  *ndata = *(int*)dataptr;
  dataptr += sizeof(int);
  *data = (void**)malloc(sizeof(void**)*(*ndata));
  size_t shift = *ndata*sizeof(size_t);
  for(int i = 0; i < *ndata; i++){
    (*data)[i] = dataptr+shift;
    data_size[i] = dataptr + i*sizeof(size_t);
    shift+=*(data_size[i]);
  }
}
