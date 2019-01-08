// Original code: http://www.kevinbeason.com/smallpt/
//
// Modifications have been made, mostly to make the algorithm as simple as possible:
//   - Some from https://github.com/randyridge/smallpt-cplusplus/blob/master/smallpt/smallpt.cpp
//   - Better style
//   - No pixel subdivision
//   - No tent filter
//   - Diffuse material only
//     This was done to allow easier GPU parallelisation. It removes most of code branching
//
// Changes for CUDA:
//   - Different random number function
//   - Spheres are declared in the function that generates the image. Then, they are copied to GPU, therefore
//     functions that need them will take take a pointer to them as a parameter

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>             // CUDA
#include <device_launch_parameters.h> // CUDA
#define M_PI 3.1415926535897932384626433832795

using namespace std;
using namespace std::chrono;

__device__ static float erand48(unsigned int *seed0, unsigned int *seed1) {
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	// Convert to float
	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

	return (res.f - 2.f) / 2.f;
}

struct Vec
{
	double x, y, z;                  // position, also color (r,g,b)
	__device__ __host__ Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
	__device__ __host__ Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	__device__ __host__ Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	__device__ __host__ Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
	__device__ __host__ Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
	__device__ __host__ Vec& norm() { return *this = *this * (1 / sqrt(x*x + y * y + z * z)); }
	__device__ __host__ double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }
	__device__ __host__ Vec operator%(Vec&b) { return Vec(y*b.z - z * b.y, z*b.x - x * b.z, x*b.y - y * b.x); } // Cross
};

struct Ray
{
	Vec o, d;
	__device__ Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

struct Sphere
{
	double rad;       // radius
	Vec p, e, c;      // position, emission, color

	__host__ Sphere(double rad_, Vec p_, Vec e_, Vec c_) :
		rad(rad_), p(p_), e(e_), c(c_) {}

	// Returns distance, 0 if no hit
	__device__ double intersect(const Ray &r) const
	{
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		Vec op = p - r.o;                                               // p is sphere center (C)
		double t, eps = 1e-4;                                           // eps is epsilon
		double b = op.dot(r.d);                                         // 1/2 from quadratic equation setup
		double det = b * b - op.dot(op) + rad * rad;                    // (b^2-4ac)/4: a=1 because ray is normalized
		if (det < 0) return 0;                                          // ray misses sphere
		else det = sqrt(det);
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0); // return smallet positive t
	}
};

__device__ __host__ inline double clamp(double x) { return x < 0 ? 0 : x>1 ? 1 : x; }

// Color for .ppm format (between 0 and 255)
inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

// Intersects ray with scene
__device__ inline bool intersect(const Ray &r, double &t, int &id, const Sphere* dev_spheres, int n_spheres)
{
	double n = 9, d, inf = t = 1e20;
	for (int i = int(n); i--;) if ((d = dev_spheres[i].intersect(r)) && d < t) { t = d; id = i; }
	return t < inf;
}

__device__ Vec radiance(Ray &r, const Sphere* dev_spheres, int n_spheres, unsigned int *s1, unsigned int *s2)
{
	Vec output = Vec();
	Vec mask = Vec(1, 1, 1);

	// Only diffuse reflection is suported, therfore we can hardcode the number of bounces
	for (int bounce = 0; bounce < 4; bounce++)
	{
		double t;                               // distance to intersection
		int id = 0;                             // id of intersected object
		if (!intersect(r, t, id, dev_spheres, n_spheres)) return Vec(); // if miss, return black
		const Sphere &obj = dev_spheres[id];    // the hit object
		Vec x = r.o + r.d * t;                  // ray intersection point
		Vec n = (x - obj.p).norm();             // sphere normal
		Vec nl = n.dot(r.d) < 0 ? n : n * -1;   // properly oriented surface normal

		output = output + mask.mult(obj.e);

		// Ideal diffuse reflection
		double r1 = 2 * M_PI * erand48(s1, s2);                                // angle around
		double r2 = erand48(s1, s2), r2s = sqrt(r2);                           // distance from center
		Vec w = nl;                                                            // w = normal
		Vec u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u; // u is perpendicular to w
		Vec d = (u*cos(r1)*r2s + v * sin(r1)*r2s + w * sqrt(1 - r2)).norm();   // d is random reflection ray

		// New ray
		r.o = x + nl * 0.05;
		r.d = d;

		mask = mask.mult(obj.c);   // multiply with colour of object       
		mask = mask * d.dot(nl);   // weigh light contribution using cosine of angle between incident light and normal
		mask = mask * 2.0;         // fudge factor
	}

	return output;
}

__global__ void generate_image_kernel(Vec *dev_c, int w, int h, int samps, const Sphere* dev_spheres, int n_spheres)
{
	// Pixel coords
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y  *blockDim.y + threadIdx.y;

	// Pixel index
	unsigned int i = (h - y - 1) * w + x;

	// seeds for random number generator
	unsigned int s1 = x, s2 = y;

	// Set up camera
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());
	Vec cx = Vec(w*.5135 / h);        // x direction increment (uses implicit 0 for y, z)
	Vec cy = (cx%cam.d).norm()*.5135; // y direction increment (note cross product)

	Vec r; // Final pixel color

	// Per each sample per pixel
	for (int s = 0; s < samps; s++)
	{
		// Compute ray direction
		Vec d = cam.d + cx * ((.25 + x) / w - .5) + cy * ((.25 + y) / h - .5);

		// Radiance
		r = r + radiance(Ray(cam.o + d * 140, d.norm()), dev_spheres, n_spheres, &s1, &s2) * (1. / samps);
	}

	dev_c[i] = Vec(clamp(r.x), clamp(r.y), clamp(r.z));
}

void generate_image(string name, int w, int h, int samps)
{
	cout << "Generating " << w << " x " << h << ", " << samps << " samples.\n";

	Sphere spheres[] = {
		Sphere(1e5,  Vec(1e5 + 1, 40.8, 81.6),   Vec(),           Vec(.50, .60, .07)), // Left
		Sphere(1e5,  Vec(-1e5 + 99, 40.8, 81.6), Vec(),           Vec(.35, .25, .85)), // Right
		Sphere(1e5,  Vec(50, 40.8, 1e5),         Vec(),           Vec(.75, .75, .75)), // Back
		Sphere(1e5,  Vec(50, 40.8, -1e5 + 170),  Vec(),           Vec()), // Front
		Sphere(1e5,  Vec(50, 1e5, 81.6),         Vec(),           Vec(.75, .75, .75)), // Bottom
		Sphere(1e5,  Vec(50, -1e5 + 81.6, 81.6), Vec(),           Vec(.75, .75, .75)), // Top
		Sphere(16.5, Vec(27, 16.5, 47),          Vec(),           Vec(1, 1, 1) * .999), // Mirror
		Sphere(16.5, Vec(73, 16.5, 78),          Vec(),           Vec(1, 1, 1) * .999), // Glass
		Sphere(600,  Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec())  // Ligth
	};

	// Set up device memory
	auto start_mem_copy_1 = system_clock::now();
	Vec* dev_c;
	cudaMalloc((void**)&dev_c, w * h * sizeof(Vec));
	cudaMemset(dev_c, 0, w * h * sizeof(Vec));
	Sphere* dev_spheres;
	cudaMalloc((void**)&dev_spheres, sizeof(Sphere) * 9);
	cudaMemcpy(dev_spheres, &spheres[0], sizeof(Sphere) * 9, cudaMemcpyHostToDevice);
	auto end_mem_copy_1 = system_clock::now();

	// Kernel execution
	auto start_kernel = system_clock::now();
	dim3 block(16, 16, 1);
	dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, 1);
	generate_image_kernel << < grid, block >> > (dev_c, w, h, samps, dev_spheres, 9);
	//dim3 nblocks(w / 16u, h / 16u);
	//dim3 nthreads(16u, 16u);
	//generate_image_kernel <<< nblocks, nthreads >>> (dev_c, w, h, samps, dev_spheres, 9);

	// Wait for kernel to complete
	cudaDeviceSynchronize();
	auto end_kernel = system_clock::now();

	// Set up host memory
	auto start_mem_copy_2 = system_clock::now();
	Vec* c = (Vec*)malloc(w * h * sizeof(Vec));
	// Transfer device -> host
	cudaMemcpy(c, dev_c, w * h * sizeof(Vec), cudaMemcpyDeviceToHost);
	auto end_mem_copy_2 = system_clock::now();
	
	duration<double> diff_kernel = end_kernel - start_kernel;
	duration<double> diff_mem_copy = (end_mem_copy_1 - start_mem_copy_1) + (end_mem_copy_2 - start_mem_copy_2);

	// Clean up resources
	cudaFree(dev_c);
	cudaFree(dev_spheres);

	// Write image to PPM file
	ofstream image(name + ".ppm", ofstream::out);
	image << "P3\n" << w << " " << h << "\n255\n";
	for (int i = 0; i < w*h; i++)
		image << toInt(c[i].x) << " " << toInt(c[i].y) << " " << toInt(c[i].z) << "\n";
	image.close();

	// Clean up host memory
	free(c);

	cout << "Done.\nTime taken for kernel: " << diff_kernel.count() << " s\nTime taken for mem copy: " << diff_mem_copy.count() << " s\n";
}

int main(int argc, char *argv[])
{
	int w = 1024, h = 768; // Resolution
	int samps = 100;       // Samples per pixel

	generate_image("image_cuda_r" + to_string(w * h) + "_s" + to_string(samps) + "_b4", w, h, samps);
}