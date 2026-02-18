#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

constexpr double correctpi = 3.14159265358979323846264338327950288419716939937510;

inline double f(const double x)
{
	return sqrt(1-x*x);
}

// simple sequential Riemann sum
double riemann_seq(const unsigned long n)
{
	const double Dx = 1.0/n;
	double sum = 0.0;
	for(unsigned long i=0; i<n; ++i)
		sum += Dx*f(i*Dx);
	return sum;
}

// OpenMP parallel version of Riemann sum
double riemann_omp(const unsigned long n)
{
	const double Dx = 1.0/n;
	double sum = 0.0;
	#pragma omp parallel // starting parallell (forking) region
	{
		double private_sum = 0.0;
		#pragma omp for nowait // distributing the chunks of the loop for the threads
		for(unsigned long i = 0; i < n; i++)
			private_sum += Dx * f(i * Dx); // doing a private sum for each thread so that we avoid synchronization over and over
		#pragma omp atomic
		sum += private_sum;
		// The end of the parallel region (here all the child-threads of the master thread are joined)
	}
	return sum;
}

// OpenMP + SSE version of Riemann sum
double riemann_omp_sse(const unsigned long n)
{
	#if defined(__SSE__)
		const double Dx = 1.0 / n;
		__m128d vec_Dx = _mm_set1_pd(Dx);
		double sum = 0.0;
		#pragma omp parallel
		{
			__m128d vec_private_sum = _mm_setzero_pd();
			#pragma omp for nowait
			for (unsigned long i = 0; i < n; i += 2) { // using sse intrinsics to compute two times per iteration
				__m128d vec_i = _mm_set_pd(i + 1, i);
				__m128d vec_x = _mm_mul_pd(vec_i, vec_Dx);
				__m128d vec_fx = _mm_sqrt_pd(_mm_sub_pd(_mm_set1_pd(1.0), _mm_mul_pd(vec_x, vec_x)));
				vec_private_sum = _mm_add_pd(vec_private_sum, _mm_mul_pd(vec_fx, vec_Dx));
			}
			double private_sum_array[2];
			_mm_storeu_pd(private_sum_array, vec_private_sum);
			double private_sum = private_sum_array[0] + private_sum_array[1];
			#pragma omp atomic
			sum += private_sum;
		}
		return sum;
		#else
			printf("SSE2 not available on this machine.\n");
			return 0.0;
		#endif
}


int main(int argc, char* argv[]) {

	const unsigned long n = strtoul(argv[1], nullptr, 0);
	const int nproc = omp_get_max_threads();
	assert(n>0 && n%2==0);

	printf("n = %lu\n", n);
	printf("nproc = %d\n", nproc);
	double pi;

	double ts = omp_get_wtime();
	pi = 4.0*riemann_seq(n);
	ts = omp_get_wtime()-ts;
	printf("seq, elapsed time = %.3f seconds, err = %.15f\n", ts, fabs(correctpi-pi));

	double tp = omp_get_wtime();
	pi = 4.0 * riemann_omp(n);
	tp = omp_get_wtime() - tp;
	printf("omp, elapsed time = %.3f seconds, err = %.15f\n", tp, fabs(correctpi - pi));

	double tsse = omp_get_wtime();
	pi = 4.0 * riemann_omp_sse(n);
	tsse = omp_get_wtime() - tsse;
	printf("omp+sse, elapsed time = %.3f seconds, err = %.15f\n", tsse, fabs(correctpi - pi));
	
	return EXIT_SUCCESS;
}
