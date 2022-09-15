#include <iostream>
#include <cu_cutoff.hpp>
#include <cutf/type.hpp>

template <class T>
void test(const std::uint32_t m, const std::uint32_t n, const std::uint32_t ld, const typename mtk::cu_cutoff::detail::base_t<T>::type threshold) {
	T* matrix_ptr;
	cudaMallocHost(&matrix_ptr, sizeof(T) * ld * n * mtk::cu_cutoff::detail::base_t<T>::num_elements);
	for (std::uint32_t i = 0; i < m * n; i++) {
		const auto v = cutf::type::cast<typename mtk::cu_cutoff::detail::base_t<T>::type>(std::pow(2.0, (static_cast<int>(i) - static_cast<int>(m * n))));
		matrix_ptr[(i % m) + (i / m) * ld] = v;
		std::printf("%e ", v);
	}
	std::printf("\n");

	cudaDeviceSynchronize();

	mtk::cu_cutoff::cutoff_small_abs_values(
			matrix_ptr,
			m, n, ld,
			threshold
			);

	cudaDeviceSynchronize();

	for (std::uint32_t i = 0; i < m * n; i++) {
		const auto v = matrix_ptr[(i % m) + (i / m) * ld];
		std::printf("%e ", v);
	}
	std::printf("\n");

	cudaFreeHost(matrix_ptr);
}

int main() {
	test<double>(5, 5, 6, 0.01);
}
