There are 3 files:

* `original.cpp`: this contains the base code that was run with the profiler
* `initial_anlysis.cpp`: contains the code modified to allow timing data gathering. It also contains OpenMP implementation
* `kernel.cu`: the CUDA version of the code

The image output is, for example, `image_sequential_r78462_s100_b4.ppm`. The number after `r` is the resolution in pixels, `s` the samples per pixel and `b` the number of times a ray bounces.