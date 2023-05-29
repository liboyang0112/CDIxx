#include <stdio.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
    Mat img = imread( argv[1], IMREAD_UNCHANGED );
    unsigned char *input = (unsigned char*)(img.data);
    printf("%d,%d,%lu\n", img.rows, img.cols,img.step*1);
    for(int j = 0;j < img.rows;j++){
        for(int i = 0;i < img.cols*3;i++){
            unsigned char &b = input[img.step * j + i ] ;
            unsigned char &g = input[img.step * j + i + 1];
            unsigned char &r = input[img.step * j + i + 2];
            //printf("%d,%d,%d\n",r,g,b);
            //exit(0);
            if(abs(b-251)<45 && abs(r-2)<45 && abs(g-88)<45){
                r = g = b = 255;
            }
        }
    }
    imwrite("image.png", img);
    return 0;
}
