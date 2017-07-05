// g++ -std=c++11 -Wall -O3 -march=native -o fast_rng fast_rng.cpp

#include <random>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include "immintrin.h" // for intrinsics
#include "fast_rng.hpp"

using namespace std;


void print_vec(float *a)
{
  for (int i=0; i < 8; i++)
    cout << a[i] << " ";
  cout << "\n\n";
}


// Traditional scalar implementation for comparison
// References:
//   https://en.wikipedia.org/wiki/Xorshift
//   https://arxiv.org/abs/1404.0390
struct xorshift_plus
{
  vector<uint64_t> seeds;

  xorshift_plus(uint64_t _s0 = rd(), uint64_t _s1 = rd(),
		uint64_t _s2 = rd(), uint64_t _s3 = rd(),
		uint64_t _s4 = rd(), uint64_t _s5 = rd(),
		uint64_t _s6 = rd(), uint64_t _s7 = rd())
    : seeds{_s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7} {};

  inline void gen_floats(float *rn)
  {
    for (int i=0; i<8; i+=2)
      {
	uint64_t x = seeds[i];
	uint64_t y = seeds[i+1];

	seeds[i] = y;
	x ^= (x << 23);
	seeds[i+1] = x ^ y ^ (x >> 17) ^ (y >> 26);

	uint64_t tmp = seeds[i+1] + y;
	uint32_t tmp0 = tmp; // low 32 bits
	uint32_t tmp1 = tmp >> 32; // high 32

	rn[i] = float(int32_t(tmp0)) * 4.6566129e-10;
	rn[i+1] = float(int32_t(tmp1)) * 4.6566129e-10;
      }
  }
};

bool test_xorshift(int niter=100)
{
  float rn1 = rd();
  float rn2 = rd();
  float rn3 = rd();
  float rn4 = rd();
  float rn5 = rd();
  float rn6 = rd();
  float rn7 = rd();
  float rn8 = rd();

  vec_xorshift_plus a(_mm256_setr_epi64x(rn1, rn3, rn5, rn7), _mm256_setr_epi64x(rn2, rn4, rn6, rn8));
  float vrn_vec[8];

  xorshift_plus b(rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8);
  float srn_vec[8];

  for (int iter=0; iter < niter; iter++)
    {
      __m256 vrn = a.gen_floats();
      _mm256_storeu_ps(&vrn_vec[0], vrn);
      b.gen_floats(srn_vec);

      for (int i=0; i<8; i++)
	{
	  if (srn_vec[i] != vrn_vec[i])
	    {
	      cout << "S code outputs: ";
	      print_vec(srn_vec);
	      cout << "V code outputs: ";
	      print_vec(vrn_vec);
	      cout << "rng test failed: scalar and vectorized prngs are out of sync!" << endl;
	      return false;
	    }
	}
    }

  cout << "All rng tests passed." << endl;
  return true;
}

// Timing helper functions
// FIXME using vintage-1970 gettimeofday().  C++11 std::chrono is supposed to be much better!
inline double time_diff(const struct timeval &tv1, const struct timeval &tv2)
{
    return (tv2.tv_sec - tv1.tv_sec) + 1.0e-6 * (tv2.tv_usec - tv1.tv_usec);
}

inline struct timeval get_time()
{
    struct timeval ret;
    if (gettimeofday(&ret, NULL) < 0)
        throw std::runtime_error("gettimeofday() failed");
    return ret;
}


static void time_mt19937(float *out, int niter, int nfreq, int nt_chunk, int stride)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dist(1.0);
    
    struct timeval tv0 = get_time();
    
    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
	    for (int it = 0; it < nt_chunk; it++)
	        out[ifreq*stride + it] = dist(rng);

    struct timeval tv1 = get_time();
    double dt = time_diff(tv0, tv1);
    double noutputs = double(niter) * double(nfreq) * double(nt_chunk);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_mt19937: ns_per_output = " << ns_per_output << endl;
}


static void time_xorshift_plus(float *out, int niter, int nfreq, int nt_chunk, int stride)
{
    xorshift_plus x;
    struct timeval tv0 = get_time();
    
    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
            for (int it = 0; it < nt_chunk; it += 8)
	        x.gen_floats(out + ifreq * stride + it);

    struct timeval tv1 = get_time();
    double dt = time_diff(tv0, tv1);
    double noutputs = double(niter) * double(nfreq) * double(nt_chunk);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_xorshift_plus: ns_per_output = " << ns_per_output << endl;
}


static void time_vec_xorshift_plus(float *out, int niter, int nfreq, int nt_chunk, int stride)
{
    vec_xorshift_plus x;

    struct timeval tv0 = get_time();

    for (int iter = 0; iter < niter; iter++)
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
            for (int it = 0; it < nt_chunk; it += 8)
	        _mm256_storeu_ps(&out[ifreq*stride + it], x.gen_floats());
  
    struct timeval tv1 = get_time();
    double dt = time_diff(tv0, tv1);
    double noutputs = double(niter) * double(nfreq) * double(nt_chunk);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_vec_xorshift_plus: ns_per_output = " << ns_per_output << endl;
}


static void time_64_bits(float *out, int niter)
{
  vec_xorshift_plus x;
  __m256 tmp;
  
  struct timeval tv0 = get_time();

  for (int iter = 0; iter < niter; iter++)
    tmp = _mm256_xor_ps(tmp, x.gen_floats());

  struct timeval tv1 = get_time();
  double dt = time_diff(tv0, tv1);
  double noutputs = double(niter) * 256 / 64;
  double ns_per_output = 1.0e9 * dt / noutputs;

  _mm256_storeu_ps(out, tmp);
  
  cout << "64 bit test: ns_per_output = " << ns_per_output << endl;
}


int main(int argc, char **argv)
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
  
    cout << "--------------------------------------------------" << endl;

    float *out = new float[8];
    memset(out, 0, 8 * sizeof(float));
    time_64_bits(out, 1000000);

    cout << "--------------------------------------------------" << endl;
  
    // Test to compare the outputs of vec_xorshift_plus and xorshift_plus
    test_xorshift();

    return 0;
}
