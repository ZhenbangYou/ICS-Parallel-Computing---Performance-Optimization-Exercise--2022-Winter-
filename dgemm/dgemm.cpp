// dgemm.cpp
// DGEMM (Double GEneral Matrix Multiplication)

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <new>
#include <tuple>

typedef double
	ElementType; // You may change this to see the influence of the data type

static constexpr int kAlignInBytes = 512 / 8;
static constexpr int kAlignInElements =
	kAlignInBytes / sizeof(ElementType); // Keep compatible with AVX-512
static constexpr int kDim = 1024;        // kDim may not be pow of 2!
static constexpr int RoundUpToMultiple(int num, int base) {
	return (num + base - 1) / base * base;
}
static constexpr int kDimRoundUp = RoundUpToMultiple(kDim, kAlignInElements);

static constexpr int kRepetitions =
	10; // Due to the flutuation of actual running time, repetitions are
		// necessary

static void DgemmBaseline(const ElementType src_a[kDimRoundUp][kDimRoundUp],
						  const ElementType src_b[kDimRoundUp][kDimRoundUp],
						  ElementType       dst[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			ElementType sum {};
			for (auto k = 0; k < kDim; k++) {
				sum += src_a[i][k] * src_b[k][j];
			}
			dst[i][j] = sum;
		}
	}
}

static void DgemmOptimized(const ElementType src_a[kDimRoundUp][kDimRoundUp],
						   const ElementType src_b[kDimRoundUp][kDimRoundUp],
						   ElementType       dst[kDimRoundUp][kDimRoundUp]) {
	// Here is what your code should appear. You may modify other parts when
	// necessary
}

static void RandomlyInitMatrix(ElementType matrix[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (int j = 0; j < kDim; j++) {
			matrix[i][j] = std::rand();
		}
	}
}

constexpr ElementType kEpsilon = static_cast<ElementType>(1e-8);

static bool MatrixEqual(const ElementType result_a[kDimRoundUp][kDimRoundUp],
						const ElementType result_b[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			if (std::abs(result_a[i][j] - result_b[i][j])
				> kEpsilon
					  * (std::abs(result_a[i][j]) + std::abs(result_b[i][j]))) {
				return false;
			}
		}
	}
	return true;
}

void PrintMatrix(const ElementType matrix[kDimRoundUp][kDimRoundUp]) {
	for (auto i = 0; i < kDim; i++) {
		for (auto j = 0; j < kDim; j++) {
			std::cout << matrix[i][j] << ' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// Helper function to verify that the optimized version is functionally
// correct.
static bool Verify(void (*dgemm_kernel_baseline)(
					   const ElementType src_a[kDimRoundUp][kDimRoundUp],
					   const ElementType src_b[kDimRoundUp][kDimRoundUp],
					   ElementType       dst[kDimRoundUp][kDimRoundUp]),
				   void (*dgemm_kernel_optimized)(
					   const ElementType src_a[kDimRoundUp][kDimRoundUp],
					   const ElementType src_b[kDimRoundUp][kDimRoundUp],
					   ElementType       dst[kDimRoundUp][kDimRoundUp]),
				   int num_trials) {
	for (auto i = 0; i < num_trials; i++) {
		auto src_a = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp]; // Not initialized yet
		auto src_b = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp]; // Not initialized yet

		auto dst_baseline = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization
		auto dst_optimized = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization

		RandomlyInitMatrix(src_a);
		RandomlyInitMatrix(src_b);

		dgemm_kernel_baseline(src_a, src_b, dst_baseline);
		dgemm_kernel_optimized(src_a, src_b, dst_optimized);

#ifdef PRINT
		PrintMatrix(dst_baseline);
		PrintMatrix(dst_optimized);
#endif

		if (MatrixEqual(dst_baseline, dst_optimized) == false) {
			// Almost forget these
			operator delete[](src_a, std::align_val_t(kAlignInBytes));
			operator delete[](src_b, std::align_val_t(kAlignInBytes));
			operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
			operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
			return false;
		}

		// Almost forget these
		operator delete[](src_a, std::align_val_t(kAlignInBytes));
		operator delete[](src_b, std::align_val_t(kAlignInBytes));
		operator delete[](dst_baseline, std::align_val_t(kAlignInBytes));
		operator delete[](dst_optimized, std::align_val_t(kAlignInBytes));
	}
	return true;
}

typedef decltype(std::chrono::high_resolution_clock::now()
				 - std::chrono::high_resolution_clock::now())
	UnitOfTime; // The unit of high_resolution_clock may vary with systems,
				// so this is necessary to keep the program cross-platform

// Return value: running time of one trial (only the running time of the kernel
// is included). Time is measured by milliseconds.
static long long MeasureTime(
	void (*dgemm_kernel)(const ElementType src_a[kDimRoundUp][kDimRoundUp],
						 const ElementType src_b[kDimRoundUp][kDimRoundUp],
						 ElementType       dst[kDimRoundUp][kDimRoundUp])) {
	UnitOfTime total_time_elapsed {};
	for (auto i = 0; i < kRepetitions; i++) {
		auto src_a = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization
		auto src_b = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization
		auto dst = new (std::align_val_t(kAlignInBytes))
			ElementType[kDimRoundUp][kDimRoundUp] {}; // Zero initialization

		RandomlyInitMatrix(src_a);
		RandomlyInitMatrix(src_b);

		auto start_time_point = std::chrono::high_resolution_clock::now();
		dgemm_kernel(src_a, src_b, dst);
		auto end_time_point = std::chrono::high_resolution_clock::now();

		total_time_elapsed += end_time_point - start_time_point;

		// Almost forget these
		operator delete[](src_a, std::align_val_t(kAlignInBytes));
		operator delete[](src_b, std::align_val_t(kAlignInBytes));
		operator delete[](dst, std::align_val_t(kAlignInBytes));
	}

	using namespace std::chrono_literals;
	return total_time_elapsed / kRepetitions / 1ms;
}

static auto CalculateSpeedup(
	void (*dgemm_kernel_baseline)(
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp]),
	void (*dgemm_kernel_optimized)(
		const ElementType src_a[kDimRoundUp][kDimRoundUp],
		const ElementType src_b[kDimRoundUp][kDimRoundUp],
		ElementType       dst[kDimRoundUp][kDimRoundUp])) {
	auto running_time_baseline  = MeasureTime(dgemm_kernel_baseline);
	auto running_time_optimized = MeasureTime(dgemm_kernel_optimized);

	return std::tuple {running_time_baseline, running_time_optimized,
					   static_cast<double>(running_time_baseline)
						   / static_cast<double>(running_time_optimized)};
}

#define DEBUG

int main() {
	std::srand(std::time(nullptr));

#ifdef DEBUG
	if (Verify(DgemmBaseline, DgemmOptimized, 10) == false) {
		std::cerr << "Functionally incorrect" << std::endl;
		return -1;
	} else {
		std::cerr << "Functionally correct" << std::endl;
	}
#endif

	auto [running_time_baseline, runnning_time_optimzed, speedup] =
		CalculateSpeedup(DgemmBaseline, DgemmOptimized);
	std::cout << "Average running time of the baseline version is "
			  << running_time_baseline << "ms" << std::endl;
	std::cout << "Average running time of the optimized version is "
			  << runnning_time_optimzed << "ms" << std::endl;
	std::cout << "The speedup is " << speedup << "x" << std::endl;
}