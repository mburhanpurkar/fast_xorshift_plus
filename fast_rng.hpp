#include <random>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include "immintrin.h" // for intrinsics
#include <stdexcept>


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
  // The compiler is supposed to ensure that the vec_xorshift_plus is always aligned, so if you
  // do get an "unaligned vec_xorshift_plus" exception, then the compiler's alignment logic
  // is being defeated somehow.  One way this can happen is when constructing a std::shared_ptr<>
  // with std::make_shared<>().  For example,
  //
  //    shared_ptr<vec_xorshift_plus> = make_shared<vec_xorshift_plus>()
  //
  // can generate either an aligned or unaligned vec_xorshift_plus.  The solution in this case
  // is to construct the shared pointer as follows:
  //
  //    shared_ptr<vec_xorshift_plus> = shared_ptr<vec_xorshift_plus> (new vec_xorshift_plus());
  
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
        s0 = _mm256_setr_epi64x(rd(), rd(), rd(), rd());
        s1 = _mm256_setr_epi64x(rd(), rd(), rd(), rd());
    };

    vec_xorshift_plus(__m256i _s0, __m256i _s1)
    {

      if (!is_aligned(&s0,32) || !is_aligned(&s1,32))
	throw std::runtime_error("Fatal: unaligned vec_xorshift_plus!  See discussion in fast_rng.hpp");

      s0 = _s0;
      s1 = _s1;
    };

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

} // namespace fast_rng

#endif // _FAST_RNG_HPP
