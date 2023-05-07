#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "utils.h"
#include "intrin-wrapper.h"
#include <immintrin.h>


// // Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

using namespace std;
using namespace cv;

namespace
{
    //! [burningship-simd]
    void simdburningship(int*pixelMatrix, int rows, int cols, const float x1, const float y1, const float scaleX, const float scaleY)
    {

        // define constants out here
        const __m256d _two = _mm256_set1_pd(2.0);
        const __m256d _four = _mm256_set1_pd(4.0);
        const __m256d _zero = _mm256_set1_pd(0.0);
        const __m256d _twofivefive = _mm256_set1_pd(255.0);
        const __m256d _sign_bit = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));

        // for getting the cr values
        const __m256d _x_scale = _mm256_set1_pd(scaleX);
        const __m256d _x1 = _mm256_set1_pd(x1);

        int max_iter = 500;


        //for (int i = 0; i < rows; i++)
        for (int i = 1438; i < 1442; i++)
        {
            // get ci (ci will hold same values across the register because we go row by row)
            // ci = i / scaleY + y1
            double y0 = i / scaleY + y1;
            __m256d ci_vec = _mm256_set1_pd(y0);



            // for (int j = 0; j < cols; j += 4)
            for (int j = 7604; j < 7612; j += 4)
            {
                __m256d _j = _mm256_set_pd(j,j+1,j+2,j+3);

                // get cr (cr will hold diff values across the register)
                // cr = j / scaleX + x1
                __m256d _cr_inter = _mm256_div_pd(_j,_x_scale);
                __m256d cr_vec = _mm256_add_pd(_cr_inter,_x1);

                // for each pixel we need to reset the values to be 0
                __m256d zr = _mm256_set1_pd(0.0); // 4 copies of 0.0
                __m256d zi = _mm256_set1_pd(0.0);
                __m256d re = _mm256_set1_pd(0.0);
                __m256d im = _mm256_set1_pd(0.0);

                double my_array0[4];
                _mm256_storeu_pd(my_array0, cr_vec);
                //printf("Vector contents: (row:%d col:%d) %f %f %f %f\n", i,j, my_array0[3], my_array0[2], my_array0[1], my_array0[0]);

                // store the num iterations taken by each pixel
                __m256d counter_vec = _mm256_set1_pd(0.0); // initially, each pixel has 0 iterations --- must be INTEGER, lets make it pd

                // try this later, maybe enable me to exit early?
                __m256d max_vec = _mm256_set1_pd(max_iter); // epi32 for INTEGER, lets try it with pd



                // Begin Algorithm
                // ESCAPE TIME LOOP
                // start the main loop for this pixel
                for (int t = 0; t < max_iter; t++) {

                    if (t > 2) {

                        printf("t %d, row: %d col: %d \n", t, i, j);

                    }

                    // first generate the stopping criteria
                    // (zr * zr + zi * zi) > 4.0
                    __m256d zr2 = _mm256_mul_pd(zr, zr);
                    __m256d zi2 = _mm256_mul_pd(zi, zi);
                    __m256d sum = _mm256_add_pd(zr2, zi2);

                    // print the SUM for debug
                    double my_array[4];
                    _mm256_storeu_pd(my_array, sum);
                    printf("SUM contents: (row:%d col:%d) %f %f %f %f\n", i,j, my_array[3], my_array[2], my_array[1], my_array[0]);
                    // ok the sum is working

                    // create a mask to check each element in the vector for the condition
                    __m256d compare = _mm256_set1_pd(4.0);
                    __m256d mask = _mm256_cmp_pd(sum, compare, _CMP_LT_OQ); // do not modify the predicate it's working
                    // returns:
                    //          -nan when sum < 4
                    //          0 when sum > 4

                    // i want:
                    //          1 when sum < 4
                    //          0 when sum > 4

                    // -nan --> 1
                    mask = _mm256_and_pd(mask, _mm256_set1_pd(1.0));
                    double my_array3[4];
                    _mm256_storeu_pd(my_array3, mask);
                    printf("MASK T1 contents: (row:%d col:%d) %f %f %f %f\n", i,j, my_array3[3], my_array3[2], my_array3[1], my_array3[0]);



                    // // // print the MASK for debug
                    // double my_array1[4];
                    // _mm256_storeu_pd(my_array1, mask);
                    // printf("MASK ORIG contents: (row:%d col:%d) %f %f %f %f\n", i,j, my_array1[3], my_array1[2], my_array1[1], my_array1[0]);


                    // check if numbers in the mask are 0.0
                    //if (_mm256_testz_pd(mask, mask)) {
                        // if [0,0,0,0], then no more iterations neeeded 
                        //break;
                    //}

                    // check if you can stop iterating early because the sums are all greater than 0 already
                    __m256d zeros = _mm256_setzero_pd();
                    __m256d cmp_result = _mm256_cmp_pd(mask, zeros, _CMP_EQ_OQ);
                    int test_all_zeros = _mm256_testc_pd(cmp_result, _mm256_set1_pd(-1.0));
                    if (test_all_zeros == 1) {
                        printf("Stopping early at t=%d.\n", t);
                        break;
                    }



                    // otherwise, add the mask to the counter vec
                    // pixels which have "escaped" by now should not accumulate more num_iterations
                    // pixels which have not "escaped" by now should +1 to the num of iterations needed to satisfy
                    counter_vec = _mm256_add_pd(counter_vec, mask);
                    // print the COUNTER AGG for debug
                    // double my_array1[4];
                    // _mm256_storeu_pd(my_array1, mask);
                    // printf("Counter contents: (row:%d col:%d) %f %f %f %f\n", i,j, my_array1[3], my_array1[2], my_array1[1], my_array1[0]);

                    // Actual Fractal Computation
                    // then, do the fractal computation for this single iteration
                    // z = abs(z * z) + c;
                    // re = zr * zr - zi * zi + cr;
                    // im = fabs(zr * zi) * 2.0 + ci;
                    re = _mm256_add_pd(_mm256_sub_pd(zr2, zi2), cr_vec);

                    __m256d im_intermediate = _mm256_mul_pd(zr,zi);
                    // https://stackoverflow.com/questions/63599391/find-absolute-in-avx#:~:text=So%20absolute%20value%20can%20be,mask%20for%20the%20sign%20bit.
                    // __m256 sign_bit = _mm256_set1_ps(-0.0f);
                    // something not working here, try this
                    // const __m256d sign_bit = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));
                    __m256d abs = _mm256_andnot_pd(_sign_bit, im_intermediate);
                    __m256d im_intermediate_2 = _mm256_mul_pd(abs,_two);
                    im = _mm256_add_pd(im_intermediate_2, ci_vec);


                    zr = re;
                    zi = im;
                }
                // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

                // ********************* Grayscale value converter
                // when you have done the maximum number of iter for the 4 pixels OR if all 4 pixels are simply greater than 4
                // use the result to convert to grayscale value


                // division can only be done in single precision i guess ?
                // __m256 counter_ps = _mm256_cvtepi32_ps(counter_vec);
                // __m256 max_ps = _mm256_cvtepi32_ps(max_vec);
                // __m256 result_ps = _mm256_div_ps(counter_ps, max_ps);
                __m256d result_pd = _mm256_div_pd(counter_vec, max_vec);

                // https://stackoverflow.com/questions/61461613/my-sse-avx-optimization-for-element-wise-sqrt-is-no-boosting-why
                __m256d values_sqrt = _mm256_sqrt_pd(result_pd);
                __m256d values_mult = _mm256_mul_pd(values_sqrt, _twofivefive);

                // then i need to round this to the nearest integer
                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_round_pd&ig_expand=6159
                __m256d grayscale_pd = _mm256_round_pd(values_mult, _MM_FROUND_TO_NEAREST_INT);

                // weird casting issue
                __m256 my_vec_ps = _mm256_castpd_ps(grayscale_pd);
                __m256i grayscale_int = _mm256_cvttps_epi32(my_vec_ps);


                // ok now that I have my values, I can store them back in the pixel matrix
                int index = i+j*rows;
                // _mm256_store_si256((__m256i*)&pixelMatrix[index], grayscale_int);

                pixelMatrix[index + 0] = int(grayscale_int[3]);
                pixelMatrix[index + 1] = int(grayscale_int[2]);
                pixelMatrix[index + 2] = int(grayscale_int[1]);
                pixelMatrix[index + 3] = int(grayscale_int[0]);

                // counter intuitive, because this would normally be i*rows + j
                // but the true fractal is actually upside down, so we flip it
                // and make it i+rows*j
                // pixelMatrix[i+j*rows] = grayscale_value;
            }

            if (i%200 == 0){
                printf("row %d/%d \n", i, rows);
            }
        }
    }


    void write_pixels_to_image_file(Mat &img, int*pixelMatrix, int rows, int cols) {
        // uses openCV Mat datatype to write the pixel values and save image to disk
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int grayscale_int = pixelMatrix[i+j*rows];
                uchar value = (uchar) grayscale_int;
                img.ptr<uchar>(i)[j] = value;

                // printf("%d ", value);
            }
        }    
    }
}



int main()
{
    // define the image dimensions
    int rows_x = 9600;
    int cols_y = 10800;

    Timer t;
    t.tic();
    // allocate memory to be used for storing pixel valuess
    int* pixelMatrix = (int*) aligned_alloc(32, rows_x * cols_y * sizeof(int));
    memset(pixelMatrix, 0, rows_x * cols_y * sizeof(int));
    printf("time to aligned malloc = %f s\n", t.toc());


    // define the bounds of the burningship fractal domain 
    float x1 = -2.2f, x2 = 2.2f;
    float y1 = -2.2f, y2 = 2.2f;

    // used for mapping the pixels to the domain
    float scaleX = cols_y / (x2 - x1); // ->  9600 / (2.2 - -2.2) ~= 2000
    float scaleY = rows_x / (y2 - y1); // ->  10800 / (2.2 - -2.2) ~= 2000


    //! [color the set of pixels in the set vs not in the set]
    t.tic();
    simdburningship(pixelMatrix, rows_x, cols_y, x1, y1, scaleX, scaleY);
    printf("time to compute SIMD version = %f s\n", t.toc());


    // Render results to image file with openCV
    Mat burningshipImg(rows_x, cols_y, CV_8U);
    write_pixels_to_image_file(burningshipImg, pixelMatrix, rows_x, cols_y);
    imwrite("burningship_simd.png", burningshipImg);

    free(pixelMatrix);

    return EXIT_SUCCESS;
}
























