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
    int mandelbrot(const float &cr, const float &ci, const int max)
    {
        // complex<float> z = c;
        float zr = 0;
        float zi = 0;
        float re = 0;
        float im = 0;


        for (int t = 0; t < max; t++)
        {

            if ((zr * zr + zi * zi) > 4.0f) {
                return t;
            }

            // z = z*z + c;
            re = zr * zr - zi * zi + cr;
            im = zr * zi * 2.0 + ci;
            zr = re;
            zi = im;
        }
        return max;
    }
    //! [mandelbrot-escape-time-algorithm]


    //! [mandelbrot-grayscale-value]
    int mandelbrotFormula(const float &cr, const float &ci, const int maxIter=500) {

        int value = mandelbrot(cr, ci, maxIter);

        if(maxIter - value == 0)
        {
            return 0;
        }

        return round(sqrt(value / (float) maxIter) * 255);
    }
    //! [mandelbrot-grayscale-value]


    //! [mandelbrot-sequential]
    void sequentialMandelbrot(int*pixelMatrix, int rows, int cols, const float x1, const float y1, const float scaleX, const float scaleY)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float x0 = j / scaleX + x1;
                float y0 = i / scaleY + y1;

                complex<float> c(x0, y0);
                float cr = x0;
                float ci = y0;

                // for each pixel get the grayscale value
                int grayscale_value = mandelbrotFormula(cr,cr);

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
    float scaleX = cols_y / (x2 - x1); // ->  5400 / (0.6 - -2.1) ~= 2000
    float scaleY = rows_x / (y2 - y1); // ->  4800 / (1.2 - -1.2) ~= 2000


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
























