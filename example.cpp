#include "fast_rng.hpp"
#include "rng_helpers.hpp"
#include <iostream>
#include <random>

using namespace std;

// Shows example usage of fast_rng.hpp
static void time_mt19937(float *out, int niter, int nfreq, int nt_chunk, int stride)
{
    random_device rd;
    mt19937 rng(rd());
    uniform_real_distribution<> dist(1.0);
    
    struct timeval tv0 = rng_helpers::get_time();
    
    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    for (int it = 0; it < nt_chunk; it++)
	        out[ifreq*stride + it] = dist(rng);

    struct timeval tv1 = rng_helpers::get_time();
    double dt = rng_helpers::time_diff(tv0, tv1);
    double noutputs = double(niter) * double(nfreq) * double(nt_chunk);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_mt19937: ns_per_output = " << ns_per_output << endl;
}


static void time_xorshift_plus(float *out, int niter, int nfreq, int nt_chunk, int stride)
{
    rng_helpers::xorshift_plus x;
    struct timeval tv0 = rng_helpers::get_time();
    
    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
            for (int it = 0; it < nt_chunk; it += 8)
	        x.gen_floats(out + ifreq * stride + it);

    struct timeval tv1 = rng_helpers::get_time();
    double dt = rng_helpers::time_diff(tv0, tv1);
    double noutputs = double(niter) * double(nfreq) * double(nt_chunk);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_xorshift_plus: ns_per_output = " << ns_per_output << endl;
}


static void time_vec_xorshift_plus(float *out, int niter, int nfreq, int nt_chunk, int stride)
{
    fast_rng::vec_xorshift_plus x;

    struct timeval tv0 = rng_helpers::get_time();

    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
            for (int it = 0; it < nt_chunk; it += 8)
	        _mm256_storeu_ps(&out[ifreq*stride + it], x.gen_floats());
  
    struct timeval tv1 = rng_helpers::get_time();
    double dt = rng_helpers::time_diff(tv0, tv1);
    double noutputs = double(niter) * double(nfreq) * double(nt_chunk);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_vec_xorshift_plus: ns_per_output = " << ns_per_output << endl;
}


int main()
{
    const int nfreq = 16384;
    const int nt_chunk = 1024;
    const int stride = 4096;
    const int niter = 10;
    
    // Comparison of mt19937, xorshift_plus, and vec_xorshift_plus
    float *buf = new float[nfreq * stride];
    memset(buf, 0, nfreq * stride * sizeof(float));
    time_mt19937(buf, niter, nfreq, nt_chunk, stride);
    time_xorshift_plus(buf, niter, nfreq, nt_chunk, stride);
    time_vec_xorshift_plus(buf, niter, nfreq, nt_chunk, stride);
}
