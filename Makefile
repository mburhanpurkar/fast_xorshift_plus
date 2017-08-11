#
# Makefile
# test_rng
#

all: test example

test: test_rng.cpp fast_rng.hpp rng_helpers.hpp
	g++ -std=c++11 -Wall -O3 -march=native -o test test_rng.cpp

example: example.cpp fast_rng.hpp rng_helpers.hpp
	g++ -std=c++11 -Wall -O3 -march=native -o example example.cpp

clean:
	rm -f *.o test example
