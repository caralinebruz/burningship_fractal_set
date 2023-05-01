#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

using namespace std;

__global__ void print_hello() {
  printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char** argv)
{
    /*
    // define the image dimensions
    int rows_x = 9600; int cols_y = 10800;

    Timer t;
    t.tic();
    // allocate memory to be used for storing pixel valuess
    int* pixelMatrix = (int*) malloc(rows_x * cols_y * sizeof(int));
    printf("time to malloc = %f s\n", t.toc());

    // define the bounds of the burningship fractal domain 
    float x1 = -2.2f, x2 = 2.2f;
    float y1 = -2.2f, y2 = 2.2f;

    // used for mapping the pixels to the domain
    float scaleX = cols_y / (x2 - x1); // ->  9600 / (2.2 - -2.2) ~= 2000
    float scaleY = rows_x / (y2 - y1); // ->  10800 / (2.2 - -2.2) ~= 2000


    //! [color the set of pixels in the set vs not in the set]
    t.tic();
    sequentialburningship(pixelMatrix, rows_x, cols_y, x1, y1, scaleX, scaleY);
    printf("time to compute basic version = %f s\n", t.toc());


    // Render results to image file with openCV
    Mat burningshipImgSequential(rows_x, cols_y, CV_8U);
    write_pixels_to_image_file(burningshipImgSequential, pixelMatrix, rows_x, cols_y);
    imwrite("burningship.png", burningshipImgSequential);
    */
    print_hello<<<3, 5>>>();
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
