#
# Makefile
# test_rng
#

all: test_rng

test_rng: test_rng.cpp fast_rng.hpp
	g++ -std=c++11 -Wall -O3 -march=native -o test_rng test_rng.cpp

clean:
	rm -f *.o test_rng
