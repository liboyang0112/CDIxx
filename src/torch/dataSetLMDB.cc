#include "dataSetLMDB.h"
#include "cdilmdb.h"

dataSetLMDB::dataSetLMDB(const char* dbDIR, torch::Device &device_, torch::TensorOptions &opt, int nc_, int row_, int col_, int ncl_, int rowlabel_, int collabel_) : device(device_), options(opt), nc(nc_),row(row_), col(col_),ncl(ncl_), rowlabel(rowlabel_), collabel(collabel_){
    nsample = initLMDB(dbDIR);
}
torch::data::Example<> dataSetLMDB::get(size_t index){
  void *data, *label;
  size_t data_size = row*col*sizeof(float), label_size;
  int idx = index;
  readLMDB(&data, &data_size, &label, &label_size, &idx);
  torch::Tensor img_tensor = torch::from_blob((float*)data, {nc, row, col}).to(device); // Channels x Height x Width
  torch::Tensor label_tensor = torch::from_blob((float*)label, {ncl, rowlabel, collabel}).to(device); // Channels x Height x Width
  return {img_tensor, label_tensor};
}
torch::optional<size_t> dataSetLMDB::size() const{
  return nsample;
};
