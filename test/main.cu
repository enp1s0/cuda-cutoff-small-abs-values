#include <iostream>
#include <cu_cutoff.hpp>
#include <cutf/type.hpp>

template <class T>
std::string get_type_name();
template <> std::string get_type_name<half   >() {return "half";};
template <> std::string get_type_name<float  >() {return "float";};
template <> std::string get_type_name<double >() {return "double";};
template <> std::string get_type_name<half2  >() {return "half2";};
template <> std::string get_type_name<float2 >() {return "float2";};
template <> std::string get_type_name<double2>() {return "double2";};

template <class T>
void test(const std::uint32_t m, const std::uint32_t n, const std::uint32_t ld, const typename mtk::cu_cutoff::detail::base_t<T>::type threshold) {
	typename mtk::cu_cutoff::detail::base_t<T>::type* matrix_ptr;
	cudaMallocHost(&matrix_ptr, sizeof(typename mtk::cu_cutoff::detail::base_t<T>::type) * ld * n * mtk::cu_cutoff::detail::base_t<T>::num_elements);
	std::printf("[%7s] ", get_type_name<T>().c_str());
	for (std::uint32_t i = 0; i < m * n; i++) {
		const auto v = cutf::type::cast<typename mtk::cu_cutoff::detail::base_t<T>::type>(std::pow(2.0, (static_cast<int>(i) - static_cast<int>(m * n))));
		matrix_ptr[(i % m) + (i / m) * ld] = v;
		std::printf("%e ", cutf::type::cast<double>(v));
	}
	std::printf("\n");

	cudaDeviceSynchronize();

	mtk::cu_cutoff::cutoff_small_abs_values(
			matrix_ptr,
			m, n, ld,
			threshold
			);

	cudaDeviceSynchronize();

	std::printf("          ");
	for (std::uint32_t i = 0; i < m * n; i++) {
		const auto v = matrix_ptr[(i % m) + (i / m) * ld];
		std::printf("%e ", cutf::type::cast<double>(v));
	}
	std::printf("\n");

	cudaFreeHost(matrix_ptr);
}

int main() {
	test<half   >(5, 5, 6, 0.01);
	test<float  >(5, 5, 6, 0.01);
	test<double >(5, 5, 6, 0.01);
	test<half2  >(5, 5, 6, 0.01);
	test<float2 >(5, 5, 6, 0.01);
	test<double2>(5, 5, 6, 0.01);
}
