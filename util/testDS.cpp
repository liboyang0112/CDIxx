#include "dataSetLMDB.h"
#include "torch/torch.h"
#define imageSize 256
using namespace std;

int main(){
  int batch_size = 2;
  const char* image_dir = "testdb";
  torch::Device device(torch::kCUDA);
  auto options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(true);
  dataSetLMDB ds(image_dir, device, options, 1, imageSize, imageSize, 1, imageSize, imageSize);
  auto mdataset = ds.map(torch::data::transforms::Stack<>());
  auto mdataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(mdataset), batch_size);
  for(auto &batch: *mdataloader){
    auto data = batch.data;
    auto target = batch.target;
    cout<<data.sizes()<<endl<<target.sizes()<<endl;
  }
}
