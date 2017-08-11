#include <cstring>
#include <random>
#include <sys/time.h>
#include <stdexcept>

#ifndef _RNG_HELPERS_HPP
#define _RNG_HELPERS_HPP

namespace rng_helpers {
  
// Traditional scalar implementation for comparison
// References:
//   https://en.wikipedia.org/wiki/Xorshift
//   https://arxiv.org/abs/1404.0390
struct xorshift_plus
{
    std::vector<uint64_t> seeds;

    xorshift_plus()
    {
        std::random_device rd;
	seeds = {rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    };

    xorshift_plus(uint64_t _s0, uint64_t _s1,
		  uint64_t _s2, uint64_t _s3,
		  uint64_t _s4, uint64_t _s5,
		  uint64_t _s6, uint64_t _s7)
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


} // namespace rng_helpers


#endif  // _RNG_HELPERS_HPP
