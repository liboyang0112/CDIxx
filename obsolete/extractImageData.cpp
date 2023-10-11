#include <stdio.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
    Mat img = imread( argv[1], IMREAD_UNCHANGED );
    uint16_t *input = (uint16_t*)(img.data);
    uint16_t prevv = 0;
    for(int j = 0;j < img.rows;j++){
      auto &val = input[j*img.cols+77];
      if(j != 0) printf("%d %hu\n", j , prevv);
      printf("%d %hu\n", j, val);
      prevv = val;
      val = -1;
    }
    imwrite("changedimage.png", img);
    return 0;
}
