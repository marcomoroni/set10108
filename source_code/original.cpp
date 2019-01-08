// Original code: http://www.kevinbeason.com/smallpt/
//
// Modifications have been made, mostly to make the algorithm as simple as possible:
//   - Some from https://github.com/randyridge/smallpt-cplusplus/blob/master/smallpt/smallpt.cpp
//   - Better style
//   - No pixel subdivision
//   - No tent filter
//   - Diffuse material only
//     This was done to allow easier GPU parallelisation. It removes most of code branching

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <string>
#include <chrono>
#include <iostream>
#define M_PI 3.1415926535897932384626433832795

using namespace std;
using namespace std::chrono;

double erand48()
{
	return (double)rand() / (double)RAND_MAX;
}

struct Vec
{
	double x, y, z;                  // position, also color (r,g,b)
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
	Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
	Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
	Vec& norm() { return *this = *this * (1 / sqrt(x*x + y * y + z * z)); }
	double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }
	Vec operator%(Vec&b) { return Vec(y*b.z - z * b.y, z*b.x - x * b.z, x*b.y - y * b.x); } // Cross
};

struct Ray
{
	Vec o, d;
	Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

struct Sphere
{
	double rad;       // radius
	Vec p, e, c;      // position, emission, color

	Sphere(double rad_, Vec p_, Vec e_, Vec c_) :
		rad(rad_), p(p_), e(e_), c(c_) {}

	// Returns distance, 0 if no hit
	double intersect(const Ray &r) const
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

//Scene: radius, position, emission, color, material
Sphere spheres[] = {
	Sphere(1e5,  Vec(1e5 + 1, 40.8, 81.6),   Vec(),           Vec(.50, .60, .07) ), // Left
	Sphere(1e5,  Vec(-1e5 + 99, 40.8, 81.6), Vec(),           Vec(.35, .25, .85) ), // Right
	Sphere(1e5,  Vec(50, 40.8, 1e5),         Vec(),           Vec(.75, .75, .75) ), // Back
	Sphere(1e5,  Vec(50, 40.8, -1e5 + 170),  Vec(),           Vec()              ), // Front
	Sphere(1e5,  Vec(50, 1e5, 81.6),         Vec(),           Vec(.75, .75, .75) ), // Bottom
	Sphere(1e5,  Vec(50, -1e5 + 81.6, 81.6), Vec(),           Vec(.75, .75, .75) ), // Top
	Sphere(16.5, Vec(27, 16.5, 47),          Vec(),           Vec(1, 1, 1) * .999), // Mirror
	Sphere(16.5, Vec(73, 16.5, 78),          Vec(),           Vec(1, 1, 1) * .999), // Glass
	Sphere(600,  Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12), Vec()              )  // Ligth
};

inline double clamp(double x) { return x < 0 ? 0 : x>1 ? 1 : x; }

// Color for .ppm format (between 0 and 255)
inline int toInt(double x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }

// Intersects ray with scene
inline bool intersect(const Ray &r, double &t, int &id)
{
	double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;)
	{
		if ((d = spheres[i].intersect(r)) && d < t)
		{
			t = d;
			id = i;
		}
	}
	return t < inf;
}

// Solves the rendering equation for diffuse material only
// This allows for less code branching, easier to parallelize on the GPU
Vec radiance(Ray &r)
{
	Vec output = Vec();
	Vec mask = Vec(1, 1, 1);

	// Only diffuse reflection is suported, therfore we can hardcode the number of bounces
	for (int bounce = 0; bounce < 4; bounce++)
	{
		double t;                               // distance to intersection
		int id = 0;                             // id of intersected object
		if (!intersect(r, t, id)) return Vec(); // if miss, return black
		const Sphere &obj = spheres[id];        // the hit object
		Vec x = r.o + r.d * t;                  // ray intersection point
		Vec n = (x - obj.p).norm();             // sphere normal
		Vec nl = n.dot(r.d) < 0 ? n : n * -1;   // properly oriented surface normal

		output = output + mask.mult(obj.e);

		// Ideal diffuse reflection
		double r1 = 2 * M_PI * erand48();                                      // angle around
		double r2 = erand48(), r2s = sqrt(r2);                                 // distance from center
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

void generate_image(string name, int w, int h, int samps)
{
	cout << "Generating " << w << " x " << h << ", " << samps << " samples.\n";

	// Set up camera
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());
	Vec cx = Vec(w*.5135 / h);        // x direction increment (uses implicit 0 for y, z)
	Vec cy = (cx%cam.d).norm()*.5135; // y direction increment (note cross product)

	Vec r; // Final pixel color

	Vec *c = new Vec[w*h]; // The image

	// Loop over all image pixels
	for (int y = 0; y < h; y++) // Loop over image rows
	{
		for (int x = 0; x < w; x++) // Loop cols
		{
			// Calculate array index
			int i = (h - y - 1) * w + x;

			// Per each sample per pixel
			r = Vec(); 
			for (int s = 0; s < samps; s++)
			{
				// Compute ray direction
				Vec d = cam.d + cx * ((.25 + x) / w - .5) + cy * ((.25 + y) / h - .5);

				// Radiance
				r = r + radiance(Ray(cam.o + d * 140, d.norm())) * (1. / samps);
			}

			// Final colour
			c[i] = Vec(clamp(r.x), clamp(r.y), clamp(r.z));
		}
	}

	// Write image to PPM file
	cout << "Writing to file.\n";
	ofstream image(name + ".ppm", ofstream::out);
	image << "P3\n" << w << " " << h << "\n255\n";
	for (int i = 0; i < w*h; i++)
		image << toInt(c[i].x) << " " << toInt(c[i].y) << " " << toInt(c[i].z) << "\n";
	image.close();

	cout << "Done.\n";
}

int main(int argc, char *argv[])
{
	int w = 1024, h = 768; // Resolution
	int samps = 100;       // Samples per pixel

	generate_image("image", w, h, samps);
}