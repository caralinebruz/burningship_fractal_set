#include <iostream>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "utils.h"
#include "intrin-wrapper.h"
#include <immintrin.h>


// Headers for intrinsics
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
        const __m256d sign_bit = _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000));


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
                // int grayscale_value = burningshipFormula(cr,ci);

                // instead of calling separate functions, do everything in here
                // it will enable me to store the vectorized result
                // wont have to worry about how to return the vectorized result??
                int max_iter = 500;

                __m256d zr = _mm256_set1_pd(0.0); // 4 copies of 0.0
                __m256d zi = _mm256_set1_pd(0.0);
                __m256d re = _mm256_set1_pd(0.0);
                __m256d im = _mm256_set1_pd(0.0);

                // make a vector to store copies of the starting c values of the loop
                __m256d cr_vec = _mm256_set1_pd(cr); // 4 copies of cr
                __m256d ci_vec = _mm256_set1_pd(ci); // 4 copies of ci

                // also make vectors to store the intermediate counter for num iterations taken
                __m256i counter_vec = _mm256_set1_epi32(0); // initially, each pixel has 0 iterations --- must be INTEGER

                // try this later, maybe enable me to exit early?
                __m256i max_vec = _mm256_set1_epi32(max_iter); // epi32 for INTEGER

                // start the main loop for this pixel
                for (int t = 0; t < max_iter; t++) {

                    // first generate the condition 
                    // (zr * zr + zi * zi) > 4.0
                    __m256d zr2 = _mm256_mul_pd(zr, zr);
                    __m256d zi2 = _mm256_mul_pd(zi, zi);
                    __m256d sum = _mm256_add_pd(zr2, zi2);

                    // create a mask to check each element in the vector for the condition
                    __m256d mask = _mm256_cmp_pd(sum, _four, _CMP_LE_OQ);
                        // sum <= 4
                        // produces [0,0,0,1] for example for (false, false, false, true)

                    // check if all of the four pixels sum's are NOT less than zero (they are all false)
                    if (_mm256_testz_pd(mask, mask)) {

                        // then what is done here? 
                        // no more iterations, the counter vec is OK to be used by the output
                        break;
                    }

                    // otherwise, add the mask to the counter vec
                    // pixels which have "escaped" by now should not accumulate more num_iterations
                    // pixels which have not "escaped" by now should +1 to the num of iterations needed to satisfy
                    counter_vec = _mm256_add_epi32(counter_vec, _mm256_castpd_si256(mask));
                    // ^ mask needs to be casted from integer to 32 bit
                    // omg, then cast back to int

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
                    __m256d abs = _mm256_andnot_pd(sign_bit, im_intermediate);
                    __m256d im_intermediate_2 = _mm256_mul_pd(abs,_two);
                    im = _mm256_add_pd(im_intermediate_2, ci_vec);
                        // come back to this

                    zr = re;
                    zi = im;
                }

                // when you have done the maximum number of iter for the 4 pixels OR if all 4 pixels are simply greater than 4
                // use the result to convert to grayscale value

                // make a mask to compare each of the four values to the max_iter
                __m256d max_minus_value = _mm256_sub_pd(max_vec, counter_vec);
                // ^ this should return a result like [499,10,1,213]

                __m256d mask_zero_grayscale = _mm256_cmp_pd(max_minus_value, _zero, _CMP_EQ_OQ);
                // ^ this should provide a result like [1,0,0,1] for (true, false, false, true)

                // if it was true, then the grayscale value should be 0
                // if it was false, then the grayscale value should be computed like the formula

                // i can try to multiply the mask by the calculation. this will keep 0's in tact
                __m256d values = _mm256_div_pd(counter_vec, max_vec);

                // https://stackoverflow.com/questions/61461613/my-sse-avx-optimization-for-element-wise-sqrt-is-no-boosting-why
                values = _mm256_sqrt_pd(values);
                __m256d values_2 = _mm256_mul_pd(values, _twofivefive);

                // then i need to round this to the nearest integer

                // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_round_pd&ig_expand=6159
                __m256d grayscale_values = _mm256_round_pd(values_2,_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);


                // ok now that I have my values, I can store them back in the pixel matrix


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
    int* pixelMatrix = (int*) aligned_alloc(rows_x * cols_y * sizeof(int));
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
    //Mat burningshipImg(rows_x, cols_y, CV_8U);
   // write_pixels_to_image_file(burningshipImg, pixelMatrix, rows_x, cols_y);
    //imwrite("burningship_simd.png", burningshipImg);

    aligned_free(pixelMatrix);

    return EXIT_SUCCESS;
}
























