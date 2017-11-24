#include <random>
#include "immintrin.h" // for intrinsics
#include <stdexcept>
#include "rng_helpers.hpp"


#ifndef _FAST_RNG_HPP
#define _FAST_RNG_HPP


#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif


namespace fast_rng {

inline bool is_aligned(const void *ptr, uintptr_t nbytes)
{

  // Note: the vec_xorshift_plus must be "aligned", in the sense that the memory addresses of
  // its 's0' and 's1' members lie on 32-byte boundaries.  Otherwise, a segfault will result!
  // To protect against this, the vec_xorshift_constructors now test for alignedness, and throw
  // an exception if unaligned.
  //
  // Generally speaking, the compiler will correctly align the vec_xorshift_plus if it is allocated
  // on the call stack of a function, but it may not be aligned if it is allocated in the heap (or
  // embedded in a larger heap-allocated class).  In this case, one solution is to represent the
  // persistent rng state as a uint64_t[8] (which can be in the heap).  When a vec_xorshift_plus is
  // needed, a temporary one can be constructed on the stack, using load/store functions (see below)
  // to exchange state with the uint64_t[8].

  // According to C++11 spec, "uintptr_t" is an unsigned integer type
  // which is guaranteed large enough to represent a pointer.
  return (uintptr_t(ptr) % nbytes) == 0;
}


// Vectorized implementation of xorshift+ using intel intrinsics
// (requires AVX2 instruction set)
struct vec_xorshift_plus
{
    // Seed values
    __m256i s0; 
    __m256i s1;

    // Initialize seeds to random device, unless alternate seeds are specified
    vec_xorshift_plus()
    {
        if (!is_aligned(&s0,32) || !is_aligned(&s1,32))
	    throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in fast_rng.hpp");
      
        std::random_device rd;
        s0 = _mm256_setr_epi64x(rng_helpers::rd64(rd), rng_helpers::rd64(rd), rng_helpers::rd64(rd), rng_helpers::rd64(rd));
	s1 = _mm256_setr_epi64x(rng_helpers::rd64(rd), rng_helpers::rd64(rd), rng_helpers::rd64(rd), rng_helpers::rd64(rd));
    };

  
    // Initialze to user-defined values (helpful for debugging)
    vec_xorshift_plus(__m256i _s0, __m256i _s1)
    {
        if (!is_aligned(&s0,32) || !is_aligned(&s1,32))
	    throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in fast_rng.hpp");

	s0 = _s0;
	s1 = _s1;
    };

    // Generates 256 random bits
    inline __m256i gen_rand_bits()
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

	return _mm256_add_epi64(y, s1);
    }

    // Interpret as eight signed 32-bit flots
    inline __m256 gen_floats()
    {
        // Convert to 8 signed 32-bit floats in range (-1, 1), since we multiply by 
        // a prefactor of 2^(-31)
        return _mm256_mul_ps(_mm256_cvtepi32_ps(gen_rand_bits()), _mm256_set1_ps(4.6566129e-10));
    }  

    //  // Generate an array of size N with random numbers on (-1, 1). 
    // inline void gen_arr(float *out, size_t N)
    // {
    //     if (N % 8 != 0)
    // 	    throw std::runtime_error("gen_arr(out, N): N must be divisible by 8!");

    // 	for (int i=0; i < N; i+=8)
    // 	  _mm256_storeu_ps(out + i, gen_floats());
    // }
  
};
 
} // namespace fast_rng

#endif // _FAST_RNG_HPP
