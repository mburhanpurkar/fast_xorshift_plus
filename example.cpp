#include "fast_rng.hpp"
#include "rng_helpers.hpp"
#include <iostream>
#include <cassert>


using namespace std;


// The following three functions use the example of filling an array of floats, 'out',
// of size 'nx' * 'ny', with adjacent frequencies separated by the 'stride' parameter
//
// The timing difference between using std::mt19937, the scalar xorshift_plus function,
// and the vectorized vec_xorshift_plus function is shown


// Example using vec_xorshift_plus
// Note that we must store the random floats generated using the _mm256_storeu_ps()
// intrinsic
static void time_vec_xorshift_plus(float *out, int niter, int ny, int nx, int stride)
{
    assert(nx % 8 == 0);
    fast_rng::vec_xorshift_plus x;

    // Start timing
    struct timeval tv0 = rng_helpers::get_time();

    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < ny; ifreq++)
            for (int it = 0; it < nx; it += 8)
	        _mm256_storeu_ps(&out[ifreq*stride + it], x.gen_floats());

    // Stop timing
    struct timeval tv1 = rng_helpers::get_time();
    double dt = rng_helpers::time_diff(tv0, tv1);
    double noutputs = double(niter) * double(ny) * double(nx);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_vec_xorshift_plus: ns_per_output = " << ns_per_output << endl;
}


static void time_xorshift_plus(float *out, int niter, int ny, int nx, int stride)
{
    assert(nx % 8 == 0);
    rng_helpers::xorshift_plus x;

    // Start timing
    struct timeval tv0 = rng_helpers::get_time();
    
    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < ny; ifreq++)
            for (int it = 0; it < nx; it += 8)
	        x.gen_floats(out + ifreq * stride + it);

    // Stop timing
    struct timeval tv1 = rng_helpers::get_time();
    double dt = rng_helpers::time_diff(tv0, tv1);
    double noutputs = double(niter) * double(ny) * double(nx);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_xorshift_plus: ns_per_output = " << ns_per_output << endl;
}



static void time_mt19937(float *out, int niter, int ny, int nx, int stride)
{
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<> dist(1.0);

    // Start timing
    struct timeval tv0 = rng_helpers::get_time();
    
    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < ny; ifreq++)
	    for (int it = 0; it < nx; it++)
	        out[ifreq*stride + it] = dist(rng);

    // Stop timing
    struct timeval tv1 = rng_helpers::get_time();
    double dt = rng_helpers::time_diff(tv0, tv1);
    double noutputs = double(niter) * double(ny) * double(nx);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_mt19937: ns_per_output = " << ns_per_output << endl;
}


int main()
{
    const int ny = 16384;
    const int nx = 1024;
    const int stride = 4096;
    const int niter = 10;
    
    // Comparison of mt19937, xorshift_plus, and vec_xorshift_plus
    float *buf = new float[ny * stride];
    memset(buf, 0, ny * stride * sizeof(float));
    time_mt19937(buf, niter, ny, nx, stride);
    time_xorshift_plus(buf, niter, ny, nx, stride);
    time_vec_xorshift_plus(buf, niter, ny, nx, stride);
}
