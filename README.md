## fast_xorshift_plus
This is a small header-only library ([fast_rng.hpp](https://github.com/mburhanpurkar/fast_xorshift_plus/blob/master/fast_rng.hpp)) for fast random number generation. Currently, it **requires the AVX2 instruction
set** (for `_mm256_xor_si256()`, `_mm256_slli_epi64()`, `_mm256_srli_epi64()`, and `_mm256_add_epi64()`). I hope to make this compatible with SSE4.2 soon! It has been optimised using [Intel Intrinsic functions](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=MMX,SSE,SSE2,SSE3,SSSE3,SSE4_1,SSE4_2,AVX,AVX2&expand=0) 
and makes use of SIMD vectorization, so the speed increase comes from running several PRNGs in parallel. 


#### Usage
To check that the generator is running properly, `make test` (see [test_rng.cpp](https://github.com/mburhanpurkar/fast_xorshift_plus/blob/master/test_rng.cpp)) that runs unit tests to compare the vectorized 
implementation to a scalar implementation (in [rng_helpers.hpp](https://github.com/mburhanpurkar/fast_xorshift_plus/blob/master/rng_helpers.hpp)). 

To use `fast_rng::gen_rand_bits()` (that generates 256 random bits) or `fast_rng::gen_floats()` (interprets the 256 bits as eight 
32-bit signed floats), use the following:

```
#include "fast_rng.hpp”

fast_rng::vec_xorshift_plus rng; // instantiate random number generator with random device

float out[8]; // array to hold the floats generated
_mm256_storeu_ps(&out, rng.gen_floats()); // store random numbers in array

__m256i rn_vec; // SIMD vector to hold bits generated
_mm256_store_si256(&rn_vec, rng.gen_rand_bits()); // store random numbers in vector
```

For a more comprehensive example using `gen_floats()` see [example.cpp](https://github.com/mburhanpurkar/fast_xorshift_plus/blob/master/example.cpp) (`make example`). For use of `gen_random_bits()`, see 
`time_64_bits()` in [test_rng.cpp](https://github.com/mburhanpurkar/fast_xorshift_plus/blob/master/test_rng.cpp). The unit test and timing test live in [test_rng.cpp](https://github.com/mburhanpurkar/fast_xorshift_plus/blob/master/test_rng.cpp), which can be compiled using `make test`. 
