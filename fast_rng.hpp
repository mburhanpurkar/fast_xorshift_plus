// g++ -std=c++11 -Wall -O3 -march=native -o fast_rng fast_rng.hpp


#include <random>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include "immintrin.h" // for intrinsics


using namespace std;
static random_device rd;


// Vectorized implementation of xorshift+ using intel intrinsics
// (requires AVX2 instruction set)
struct vec_xorshift_plus
{
    // Seed values
    __m256i s0; 
    __m256i s1;

    // Initialize seeds to random device, unless alternate seeds are specified
    vec_xorshift_plus(__m256i _s0 = _mm256_setr_epi64x(rd(), rd(), rd(), rd()), __m256i _s1 = _mm256_setr_epi64x(rd(), rd(), rd(), rd())) : s0{_s0}, s1{_s1} {};


    // Generates 256 random bits (interpreted as 8 signed floats)
    // Returns an __m256 vector, so bits must be stored using _mm256_storeu_ps() intrinsic!
    inline __m256 gen_floats()
    {
        // x = s0
        __m256i x = s0;
        // y = s1
        __m256i y = s1;
        // s0 = y
        s0 = y;
        // x ^= (x << 23)
        x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 23));
        // s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
        s1 = _mm256_xor_si256(x, y);
        s1 = _mm256_xor_si256(s1, _mm256_srli_epi64(x, 17));
        s1 = _mm256_xor_si256(s1, _mm256_srli_epi64(y, 26));
      
        // Convert to 8 signed 32-bit floats in range (-1, 1), since we multiply by 
        // a prefactor of 2^(-31)
        return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi64(y, s1)), _mm256_set1_ps(4.6566129e-10));
    }
};


// Traditional scalar implementation for comparison
// References:
//   https://en.wikipedia.org/wiki/Xorshift
//   https://arxiv.org/abs/1404.0390
struct xorshift_plus
{
    uint64_t s0;
    uint64_t s1;

    xorshift_plus(uint64_t _s0 = rd(), uint64_t _s1 = rd()) : s0{_s0}, s1{_s1} {};
  
    inline void gen_floats(float &v1, float &v2)
    {
        uint64_t x = s0;
        uint64_t y = s1;

	s0 = y;
	x ^= (x << 23);
	s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
	
	uint64_t tmp = s1 + y;
	uint32_t tmp0 = tmp; // low 32 bits
	uint32_t tmp1 = tmp >> 32; // high 32
	
	v1 = float(int32_t(tmp0)) * 4.6566129e-10;
	v2 = float(int32_t(tmp1)) * 4.6566129e-10;
    }
};


bool test_xorshift(uint64_t i = 3289321, uint64_t j = 4328934)
{
    vec_xorshift_plus x(_mm256_set1_epi64x(i), _mm256_set1_epi64x(j));
    float rn_vec[8];
    __m256 vrn = x.gen_floats();
    _mm256_storeu_ps(&rn_vec[0], vrn);
    
    for (int i=2; i<8; i+=2)
    {
        if (rn_vec[i] != rn_vec[0] || rn_vec[i+1] != rn_vec[1])
	{
	    cout << rn_vec[i] << " should equal " <<  rn_vec[0] << " and " << rn_vec[i+1] << " should equal " << rn_vec[1] << endl;
	    cout << "vec_xorshift_plus failed: vector elements unequal!" << endl;
	    return false;
	}
    }

    xorshift_plus y(i, j);
    float srn1, srn2;
    y.gen_floats(srn1, srn2);
    
    if (srn1 == rn_vec[0] && srn2 == rn_vec[1])
    {
        cout << "test_xorshift passed!" << endl;
	return true;
    }

    cout << "test_xorshift failed -- something's wrong!" << endl;
    return false;
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
    float rn1, rn2;
    struct timeval tv0 = get_time();
    
    for (int iter = 0; iter < niter; iter++)
    {
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
	{
            for (int it = 0; it < nt_chunk; it += 2)
	    {
	        x.gen_floats(rn1, rn2);
		out[ifreq*stride + it] = rn1;
		out[ifreq*stride + it + 1] = rn2;
	    }
	}
    }

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
    {
        for (int ifreq = 0; ifreq < nfreq; ifreq++)
	{
            for (int it = 0; it < nt_chunk; it += 8)
	        _mm256_storeu_ps(&out[ifreq*stride + it], x.gen_floats());
	}
    }
  
    struct timeval tv1 = get_time();
    double dt = time_diff(tv0, tv1);
    double noutputs = double(niter) * double(nfreq) * double(nt_chunk);
    double ns_per_output = 1.0e9 * dt / noutputs;

    cout << "time_vec_xorshift_plus: ns_per_output = " << ns_per_output << endl;
}


// -------------------------------------------------------------------------------------------------
// Uncomment for testing!
// int main(int argc, char **argv)
// {
//     const int nfreq = 16384;
//     const int nt_chunk = 1024;
//     const int stride = 4096;
//     const int niter = 10;
    
//     // Comparison of mt19937, xorshift_plus, and vec_xorshift_plus
//     float *buf = new float[nfreq * stride];
//     memset(buf, 0, nfreq * stride * sizeof(float));
//     time_mt19937(buf, niter, nfreq, nt_chunk, stride);
//     time_xorshift_plus(buf, niter, nfreq, nt_chunk, stride);
//     time_vec_xorshift_plus(buf, niter, nfreq, nt_chunk, stride);
  
//     cout << "--------------------------------------------------" << endl;
  
//     // Test to compare the outputs of vec_xorshift_plus and xorshift_plus
//     test_xorshift();

//     return 0;
// }
