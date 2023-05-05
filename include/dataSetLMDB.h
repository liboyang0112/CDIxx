#include "torch/data/datasets/base.h"

class dataSetLMDB:public torch::data::datasets::Dataset<dataSetLMDB>{
public:
    dataSetLMDB(const char* dataSetDir, torch::Device &device, torch::TensorOptions &opt, int nc_, int row_, int col_, int ncl_, int rowl_, int coll_);
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;    // Override size() function, return the length of data
    torch::optional<size_t> size() const override;
    torch::TensorOptions &options;
private:
    int nc, row, col, ncl, rowlabel, collabel, nsample;
    torch::Device &device;
};
