#define _GLIBCXX_USE_CXX11_ABI 0

#include <iostream>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

#include "utils.h"

void write_pixels_to_image_file(Mat &img, int*pixelMatrix, int rows, int cols) {
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
  sequentialburningship(pixelMatrix, rows_x, cols_y, x1, y1, scaleX, scaleY);
  printf("time to compute basic version = %f s\n", t.toc());

  // Render results to image file with openCV
  Mat burningshipImgSequential(rows_x, cols_y, CV_8U);
  write_pixels_to_image_file(burningshipImgSequential, pixelMatrix, rows_x, cols_y);
  imwrite("burningship.png", burningshipImgSequential);

  delete[] pixelMatrix;
}