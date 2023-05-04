// load the following modules to render image
// module load cuda-11.8

#define _GLIBCXX_USE_CXX11_ABI 0

#include <iostream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace std;
using namespace cv;

// origianl module load requires
// module load gcc-12.2

namespace
{
  //! [escape-time-algorithm]
  // pixels that should be included in the set take a lot of iterations
  // pixels exluded from the set take few iterations ie. they "escape" quickly
  int burningship(const float &cr, const float &ci, const int max)
  {
    float zr = 0;
    float zi = 0;
    float re = 0;
    float im = 0;

    for (int t = 0; t < max; t++)
    {
      if ((zr * zr + zi * zi) > 4.0f) {
          return t;
      }

      // z = abs(z*z) + c;
      re = zr * zr - zi * zi + cr;
      im = fabs(zr * zi) * 2.0 + ci;

      zr = re;
      zi = im;
    }

    return max;
  }

  //! [burningship-grayscale-value]
  // converts the number of iterations taken, to be a grayscale value
  int burningshipFormula(const float &cr, const float &ci, const int maxIter=500)
  {
    int value = burningship(cr, ci, maxIter);

    if(maxIter - value == 0) {
      return 0;
    }

    int grayscale_val = std::round(sqrt(value / (float) maxIter) * 255);
    return grayscale_val;
  }

  //! [burningship-sequential]
  void sequentialburningship(int*pixelMatrix, int rows, int cols, const float x1, const float y1, const float scaleX, const float scaleY)
  {
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        // for each pixel in our image, figure out what coords that pixel
        // corresponds to in the domain of our problem
        float x0 = j / scaleX + x1;
        float y0 = i / scaleY + y1;

        // real is the x-axis
        float cr = x0;
        // imaginary is the y-axis
        float ci = y0;

        // get the grayscale value
        int grayscale_value = burningshipFormula(cr,ci);

        // counter intuitive, because this would normally be i*rows + j
        // but the true fractal is actually upside down, so we flip it
        // and make it i+rows*j
        pixelMatrix[i+j*rows] = grayscale_value;
      }

      if (i%200 == 0) {
          printf("row %d/%d \n", i, rows);
      }
    }
  }

   __global__
   void parallelburningship(int* pixelMatrix, const int rows, const int cols, const float x1, const float y1, const float scaleX, const float scaleY, const int maxIter)
   {
     int i = blockIdx.y * blockDim.y + threadIdx.y;
     int j = blockIdx.x * blockDim.x + threadIdx.x;
     //int idx = j + rows * i;
     //int idx = j + i * cols; // works, but not exact, block_dim 16
     int idx = i + j * rows; // exact!, block_dim 16

     //printf("i: %d, j: %d, idx: %d\n", i, j, idx);
   
     if (i >= rows || j >= cols) {
         return;
     }

     float x0 = j / scaleX + x1;
     float y0 = i / scaleY + y1;

     float cr = x0;
     float ci = y0;

     int value = 0;
     int grayscale_value = 0;

     float zr = 0;
     float zi = 0;
     float re = 0;
     float im = 0;
 
     for (int t = 0; t < maxIter; ++t)
     {
       if ((zr * zr + zi * zi) > 4.0f) {
           value = t;
           break;
       }
 
       // z = abs(z*z) + c;
       re = zr * zr - zi * zi + cr;
       im = fabs(zr * zi) * 2.0 + ci;
 
       zr = re;
       zi = im;
     }
 
     if (value == maxIter) {
       grayscale_value = 0;
     } else {
       //printf("color!\n");
       grayscale_value = std::round(sqrt(value / (float) maxIter) * 255);
     }

      pixelMatrix[idx] = grayscale_value;

      /*
      if(idx % 2 == 0) {
        pixelMatrix[idx] = 0;
      } else {
        pixelMatrix[idx] = 255;
      }
      */

      /*
      if (idx % 200 == 0){
        printf("idx %d/%d \n", idx, rows * cols);
      }
      */
   }

  void write_pixels_to_image_file(Mat &img, int*pixelMatrix, int rows, int cols) {
    // uses openCV Mat datatype to write the pixel values and save image to disk
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        int grayscale_int = pixelMatrix[i+j*rows];
        uchar value = (uchar) grayscale_int;
        img.ptr<uchar>(i)[j] = value;
      }
    }
  }
}

#define BLOCK_DIM 32

int main() {
  // define the image dimensions
  //int rows_x = 8;
  //int cols_y = 9;
  //int rows_x = 32; 
  //int cols_y = 36;
  int rows_x = 9600;
  int cols_y = 10800;
  int maxIter = 500;
  int total_img_pix = rows_x * cols_y;

  Timer t; t.tic();
  // allocate memory to be used for storing pixel valuess
  int* pixelMatrix     = (int*) malloc(total_img_pix * sizeof(int));
  int* pixelMatrix_out = (int*) malloc(total_img_pix * sizeof(int));
  int* d_pixelMatrix_gpu;
  
  // define the bounds of the burningship fractal domain 
  float x1 = -2.2f, x2 = 2.2f;
  float y1 = -2.2f, y2 = 2.2f;

  // used for mapping the pixels to the domain
  float scaleX = cols_y / (x2 - x1); // ->  9600 / (2.2 - -2.2) ~= 2000
  float scaleY = rows_x / (y2 - y1); // ->  10800 / (2.2 - -2.2) ~= 2000

  cudaMalloc((void**)&d_pixelMatrix_gpu, total_img_pix * sizeof(int));
  cudaMemcpy(d_pixelMatrix_gpu, pixelMatrix, total_img_pix*sizeof(int), cudaMemcpyHostToDevice);
  printf("time to malloc = %f s\n", t.toc());

  dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);
  dim3 dimGrid((int)ceil(rows_x/threadsPerBlock.x),(int)ceil(cols_y/threadsPerBlock.y));

  t.tic();
  //parallelburningship<<<total_img_pix / 1024 + 1, total_img_pix / 1024>>> (d_pixelMatrix_gpu, rows_x, cols_y, x1, y1, scaleX, scaleY, maxIter);
  parallelburningship<<<dimGrid, threadsPerBlock>>> (d_pixelMatrix_gpu, rows_x, cols_y, x1, y1, scaleX, scaleY, maxIter);
  //parallelburningship<<<rows_x, cols_y>>>(d_pixelMatrix_gpu, rows_x, cols_y, x1, y1, scaleX, scaleY, maxIter);
  cudaMemcpyAsync(pixelMatrix_out, d_pixelMatrix_gpu, sizeof(int) * total_img_pix, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("time to compute gpu version = %f s\n", t.toc());
  
  // Render results to image file with openCV
  Mat burningImgGPU(rows_x, cols_y, CV_8U);
  write_pixels_to_image_file(burningImgGPU, pixelMatrix_out, rows_x, cols_y);
  imwrite("burningship_gpu.png", burningImgGPU);

  return EXIT_SUCCESS;
}
