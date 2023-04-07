# mandelbrot_starter_code


Bare bones implementation of mandelbrot set calculation over a grid. 

At the end, uses openCV library to write result to image file. I installed this on my mac using `brew install opencv`. See Makefile for compiling.

## Algorithm
A simple algorithm to generate a representation of the Mandelbrot set is called the “escape time algorithm”. For each pixel in the rendered image, we test using the recurrence relation if the complex number is bounded or not under a maximum number of iterations. Pixels that do not belong to the Mandelbrot set will escape quickly whereas we assume that the pixel is in the set after a fixed maximum number of iterations. A high value of iterations will produce a more detailed image but the computation time will increase accordingly. We use the number of iterations needed to “escape” to depict the pixel value in the image.


### References
https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/page_tutorial_how_to_use_OpenCV_parallel_for_.html

https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/core/how_to_use_OpenCV_parallel_for_/how_to_use_OpenCV_parallel_for_.cpp

https://github.com/OneLoneCoder/Javidx9/blob/master/PixelGameEngine/SmallerProjects/OneLoneCoder_PGE_Mandelbrot.cpp
