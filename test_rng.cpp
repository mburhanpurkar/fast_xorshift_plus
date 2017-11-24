#include "fast_rng.hpp"
#include "rng_helpers.hpp"
#include <iostream>
#include <unistd.h>


using namespace std;



// Helper function for debug
inline void print_vec(float *a)
{
    for (int i=0; i < 8; i++)
      cout << a[i] << " ";
    cout << "\n\n";
}


// Randomized unit test comparing scalar and vectorized implementations
inline bool test_xorshift(int niter=10000)
{
    random_device rd;
    uint64_t rn1 = fast_rng::rd64(rd);
    uint64_t rn2 = fast_rng::rd64(rd);
    uint64_t rn3 = fast_rng::rd64(rd);
    uint64_t rn4 = fast_rng::rd64(rd);
    uint64_t rn5 = fast_rng::rd64(rd);
    uint64_t rn6 = fast_rng::rd64(rd);
    uint64_t rn7 = fast_rng::rd64(rd);
    uint64_t rn8 = fast_rng::rd64(rd);
          
    fast_rng::vec_xorshift_plus a(_mm256_setr_epi64x(rn1, rn3, rn5, rn7), _mm256_setr_epi64x(rn2, rn4, rn6, rn8));
    float vrn_vec[8];
    rng_helpers::xorshift_plus b(rn1, rn2, rn3, rn4, rn5, rn6, rn7, rn8);
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
		cout << "rng test failed: scalar and vectorized prngs are out of sync on iter " << i << "!" << endl;
		return false;
	    }
	}
    }

    cout << "All rng tests passed." << endl;
    return true;
}


// Timing test -- how long does it take to generate 64 random bit using vec_xorshift_plus?
inline void time_64_bits(__m256i *out, int niter)
{
  fast_rng::vec_xorshift_plus x;
    __m256i tmp;
  
    struct timeval tv0 = rng_helpers::get_time();
  
    for (int iter = 0; iter < niter; iter++)
        tmp = _mm256_xor_si256(tmp, x.gen_rand_bits());

    struct timeval tv1 = rng_helpers::get_time();
    double dt = rng_helpers::time_diff(tv0, tv1);
    double noutputs = double(niter) * 256; // generate 256 random bits per iteration
    double ns_per_output = 1.0e9 * dt / noutputs * 64; // actually, get the time for one bit, then mutiply by 64
    
    _mm256_store_si256(out, tmp);
  
    cout << "64 bit test: time for 64 bits = " << ns_per_output << endl;
}


// Note: #include <unistd.h> needed for usleep()
// Thanks for solving this mystery, Kendrick! 
void warm_up_cpu()
{
    // A throwaway computation which uses the CPU for ~10^9
    // clock cycles.  The details (usleep, xor) are to prevent the
    // compiler from optimizing it out!
    //
    // Empirically, this makes timing results more stable (without it,
    // the CPU seems to run slow for the first ~10^9 cycles or so.)

    long n = 0;
    for (long i = 0; i < 1000L * 1000L * 1000L; i++)
        n += (i ^ (i-1));
    usleep(n % 2);
}


int main()
{
    // Warm up the CPU - needed for timing
    warm_up_cpu();

    // Timing test
    __m256i out;
    time_64_bits(&out, 1000000);

    cout << "--------------------------------------------------" << endl;
  
    // Test to compare the outputs of vec_xorshift_plus and xorshift_plus
    test_xorshift();

    return 0;
}
