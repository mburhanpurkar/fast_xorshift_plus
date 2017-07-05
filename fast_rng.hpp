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
