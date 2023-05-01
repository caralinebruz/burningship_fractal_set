#define _GLIBCXX_USE_CXX11_ABI 0

#include <iostream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

#include "utils.h"

using namespace std;
using namespace cv;

namespace
{
  //! [escape-time-algorithm]
  // pixels that should be included in the set take a lot of iterations
  // pixels exluded from the set take few iterations ie. they "escape" quickly
  __device__ int burningship(const float &cr, const float &ci, const int max)
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
  __device__ int burningshipFormula(const float &cr, const float &ci, const int maxIter=500) {

      int value = burningship(cr, ci, maxIter);

      if(maxIter - value == 0)
      {
          return 0;
      }

      int grayscale_val = ::roundf(sqrt(value / (float) maxIter) * 255);
      return grayscale_val;
  }

  __global__ void parallelburningship(int *pixelMatrix, const int rows, const int cols, const float x1, const float y1, const float scaleX, const float scaleY, const int maxIter)
  {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= rows || j >= cols) {
        return;
    }

    float x0 = j / scaleX + x1;
    float y0 = i / scaleY + y1;

    float cr = x0;
    float ci = y0;

    int grayscale_value = burningship(cr, ci, maxIter);

    pixelMatrix[i + j * rows] = grayscale_value;
  }

  //! [burningship-sequential]
  __global__ void sequentialburningship(int* pixelMatrix, int rows, int cols, const float x1, const float y1, const float scaleX, const float scaleY)
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

      if (i%200 == 0){
        printf("row %d/%d \n", i, rows);
      }
    }
  }

  void write_pixels_to_image_file(cv::Mat &img, int*pixelMatrix, int rows, int cols) {
  // uses openCV Mat datatype to write the pixel values and save image to disk
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      int grayscale_int = pixelMatrix[i+j*rows];
      uchar value = (uchar) grayscale_int;
      img.ptr<uchar>(i)[j] = value;
    }
  }    
}
}

int main(int argc, char** argv)
{
  // define the image dimensions
  int rows_x = 9600; int cols_y = 10800;
  int maxIter = 500;

  Timer t;
  t.tic();

  int* pixelMatrix = (int*) malloc(rows_x * cols_y * sizeof(int));
  int* pixelMatrix_out = (int*) malloc(rows_x * cols_y * sizeof(int));
  int* d_pixelMatrix;

  float x1 = -2.2f, x2 = 2.2f;
  float y1 = -2.2f, y2 = 2.2f;

  float scaleX = cols_y / (x2 - x1); // ->  9600 / (2.2 - -2.2) ~= 2000
  float scaleY = rows_x / (y2 - y1); // ->  10800 / (2.2 - -2.2) ~= 2000

  //! [color the set of pixels in the set vs not in the set]
  t.tic();
  sequentialburningship<<<1,1>>>(pixelMatrix, rows_x, cols_y, x1, y1, scaleX, scaleY);
  printf("time to compute basic version = %f s\n", t.toc());

  // Render results to image file with openCV
  Mat burningshipImgSequential(rows_x, cols_y, CV_8U);
  write_pixels_to_image_file(burningshipImgSequential, pixelMatrix, rows_x, cols_y);
  imwrite("burningship.png", burningshipImgSequential);

  // Allocate device memory for pixelMatrix
  cudaMalloc((void**)&d_pixelMatrix, sizeof(int) * rows_x * cols_y);

  // Transfer data from host to device memory
  cudaMemcpy(d_pixelMatrix, pixelMatrix, sizeof(float) * rows_x * cols_y, cudaMemcpyHostToDevice);
  printf("time to copy memory to device = %f s\n", t.toc());

  t.tic();
  // Run the burningship algorithm on the GPU
  parallelburningship<<<1, 1>>>(d_pixelMatrix, rows_x, cols_y, x1, y1, scaleX, scaleY, maxIter);
  cudaDeviceSynchronize();
  printf("gpu execution = %f s\n", t.toc());

  // Transfer data back to host memory
  cudaMemcpy(pixelMatrix_out, d_pixelMatrix, sizeof(float) * rows_x * cols_y, cudaMemcpyDeviceToHost);

  // Convert pixel matrix to OpenCV Mat
  Mat img(rows_x, cols_y, CV_8UC1);
  write_pixels_to_image_file(img, pixelMatrix_out, rows_x, cols_y);

  // Save image to disk
  imwrite("burningship_gpu.png", img);

  // Free memory
  delete[] pixelMatrix;
  delete[] pixelMatrix_out;
  cudaFree(d_pixelMatrix);

  return 0;
}
