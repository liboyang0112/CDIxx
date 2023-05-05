#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h> 
#include <memory.h>
#include "lmdb.h"
#include <zlib.h>

#define E(expr) CHECK((rc = (expr)) == MDB_SUCCESS, #expr)
#define RES(err, expr) ((rc = expr) == (err) || (CHECK(!rc, #expr), 0))
#define CHECK(test, msg) ((test) ? (void)0 : ((void)fprintf(stderr, \
        "%s:%d: %s: %s\n", __FILE__, __LINE__, msg, mdb_strerror(rc)), abort()))

int i = 0, j = 0, rc;
MDB_env *env;
MDB_dbi dbi;
MDB_val key, data_LMDB;
MDB_txn *txn;
MDB_stat mst;
size_t data_cache_size = 0;
size_t mvdata_size = 0;
void* data_cache = 0;

int initLMDB(const char* dbDIR){
  mkdir(dbDIR, S_IRWXU | S_IRWXG | S_IRWXO);
  E(mdb_env_create(&env));
  E(mdb_env_set_maxreaders(env, 1));
  E(mdb_env_set_mapsize(env, 1048576000));
  E(mdb_env_open(env, dbDIR, MDB_WRITEMAP/*|MDB_NOSYNC*/, 0664));
  E(mdb_env_stat(env, &mst));
  E(mdb_txn_begin(env, NULL, /*MDB_WRITEMAP*/0, &txn));
  E(mdb_dbi_open(txn, NULL, 0, &dbi));
  return mst.ms_entries;
}
int fillLMDBCoded(int *keyval, void* data, size_t data_size)
{
  key.mv_size = sizeof(int);
  key.mv_data = keyval;
  if(mvdata_size < data_size){
    mvdata_size = data_size;
    if(data_LMDB.mv_data) free(data_LMDB.mv_data);
    data_LMDB.mv_data = malloc(data_size);
  }
  data_LMDB.mv_size = mvdata_size;
  int ec = compress(data_LMDB.mv_data, &data_LMDB.mv_size, data, data_size);
  if(ec < Z_OK){
    fprintf(stderr,"compress failed with ec %d\n", ec);
    exit(0);
  }
  ec =  mdb_put(txn, dbi, &key, &data_LMDB,0);
  if(ec!=MDB_SUCCESS){
    fprintf(stderr,"mdb_put failed with ec %d\n", ec);
    exit(0);
  }
  return ec;
}
int fillLMDB(int *keyval, void* data, size_t data_size, void* label, size_t label_size)
{
  int totalSize = sizeof(size_t)+data_size+label_size;
  if(totalSize > data_cache_size){
    data_cache_size = totalSize;
    free(data_cache);
    data_cache = malloc(data_cache_size);
  }
  memcpy(data_cache, &data_size, sizeof(size_t));
  memcpy(data_cache+sizeof(size_t), data, data_size);
  memcpy(data_cache+sizeof(size_t)+data_size, label, label_size);
  return fillLMDBCoded(keyval, data_cache, totalSize);
}

int saveLMDB(){
  E(mdb_txn_commit(txn));
  mdb_dbi_close(env, dbi);
  E(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
  E(mdb_dbi_open(txn, NULL, 0, &dbi));
}
void readLMDB(void** data, size_t* data_size, void** label, size_t *label_size, int *keyval){
  if(keyval){
    key.mv_size = sizeof(int);
    key.mv_data = keyval;
  }
  if((rc = mdb_get(txn, dbi , &key, &data_LMDB)) != MDB_SUCCESS){
    if(rc==MDB_NOTFOUND){
      fprintf(stderr,"data not found with key %d\n",*keyval);
    }
    fprintf(stderr,"data read failed with EC %d\n",rc);
    exit(0);
  }
  size_t allocedsize = *data_size;
  if(data_cache_size == 0) {
    data_cache_size = *data_size+*label_size;
    free(data_cache);
    data_cache = malloc(data_cache_size);
  }
  int err;
  while((err = uncompress(data_cache, &data_cache_size, data_LMDB.mv_data, data_LMDB.mv_size)) != Z_OK){
    if(err != Z_BUF_ERROR) {
      fprintf(stderr,"zip error: %d\n", err);
      exit(0);
    }
    data_cache_size *= 1.2;
    free(data_cache);
    data_cache = malloc(data_cache_size);
  }
  memcpy(data_size, data_cache, sizeof(size_t));
  if(*data_size > allocedsize) {
    fprintf(stderr,"memory allocated for data is not enough : %lu < %lu\n", allocedsize, *data_size);
    exit(0);
  }
  *label_size = data_LMDB.mv_size - sizeof(size_t) - *data_size;
  *data = data_cache+sizeof(size_t);
  *label = data_cache+sizeof(size_t)+*data_size;
}
