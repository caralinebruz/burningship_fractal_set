#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "utils.h"

using namespace std;
using namespace cv;

namespace
{
    //! [mandelbrot-escape-time-algorithm]
    int mandelbrot(const complex<float> &z0, const int max)
    {
        complex<float> z = z0;
        for (int t = 0; t < max; t++)
        {

            if (z.real()*z.real() + z.imag()*z.imag() > 4.0f) {
                return t;
            }

            z = z*z + z0;
        }
        return max;
    }
    //! [mandelbrot-escape-time-algorithm]


    //! [mandelbrot-grayscale-value]
    int mandelbrotFormula(const complex<float> &z0, const int maxIter=500) {
        int value = mandelbrot(z0, maxIter);
        if(maxIter - value == 0)
        {
            return 0;
        }

        return round(sqrt(value / (float) maxIter) * 255);
    }
    //! [mandelbrot-grayscale-value]


    //! [mandelbrot-sequential]
    // void sequentialMandelbrot(Mat &img, int*pixelMatrix, int rows, int cols, const float x1, const float y1, const float scaleX, const float scaleY)
    void sequentialMandelbrot(int*pixelMatrix, int rows, int cols, const float x1, const float y1, const float scaleX, const float scaleY)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float x0 = j / scaleX + x1;
                float y0 = i / scaleY + y1;

                complex<float> z0(x0, y0);

                int grayscale_value = mandelbrotFormula(z0);
                // printf("grayscale_value %d \n", grayscale_value);

                // add it to the correct location in the pixel matrix
                pixelMatrix[i+j*rows] = grayscale_value;

            }

            if (i%200 == 0){
                printf("row %d/%d \n", i, rows);
            }
        }
    }
    //! [mandelbrot-sequential]


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
}



int main()
{
    int rows_x = 4800;
    int cols_y = 5400;

    Timer t;
    t.tic();
    // allocate memory to be used for storing pixel valuess
    int* pixelMatrix = (int*) malloc(rows_x * cols_y * sizeof(int));
    printf("time to malloc = %f s\n", t.toc());


    // define the grid 
    float x1 = -2.1f, x2 = 0.6f;
    float y1 = -1.2f, y2 = 1.2f;
    float scaleX = cols_y / (x2 - x1); // ->  5400 / (0.6 - -2.1) ~= 2000.
    float scaleY = rows_x / (y2 - y1); // ->  4800 / (1.2 - -1.2) ~= 2000.


    //! [mandelbrot-transformation]
    t.tic();
    sequentialMandelbrot(pixelMatrix, rows_x, cols_y, x1, y1, scaleX, scaleY);
    printf("time to compute basic version = %f s\n", t.toc());


    // Render results to image file with openCV
    Mat mandelbrotImgSequential(rows_x, cols_y, CV_8U);
    write_pixels_to_image_file(mandelbrotImgSequential, pixelMatrix, rows_x, cols_y);
    imwrite("Mandelbrot_sequential_t.png", mandelbrotImgSequential);


    return EXIT_SUCCESS;
}
























