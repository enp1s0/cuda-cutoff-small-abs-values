#include <cu_cutoff.hpp>
#include <cutf/math.hpp>
#include <cutf/type.hpp>

namespace {
template <class T>
__global__ void cutoff_kernel(
		T* const ptr,
		const std::uint32_t m,
		const std::uint32_t n,
		const std::uint32_t ld,
		const T threshold
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= m * n) {
		return;
	}

	const auto index = (tid % m) + (tid / m) * static_cast<std::uint64_t>(ld);
	const auto v = ptr[index];
	if (cutf::math::abs(v) < threshold) {
		ptr[index] = cutf::type::cast<T>(0);
	}
}
} // unnamed namespace

template <class T>
void mtk::cu_cutoff::cutoff_small_abs_values(
		T* const ptr,
		const std::uint32_t m,
		const std::uint32_t n,
		const std::uint32_t ld,
		const typename mtk::cu_cutoff::detail::base_t<T>::type threshold,
		cudaStream_t cuda_stream
		) {
	const std::size_t block_size = 256;
	const auto grid_size = (m * n * detail::base_t<T>::num_elements + block_size - 1) / block_size;

	cutoff_kernel<typename mtk::cu_cutoff::detail::base_t<T>::type><<<grid_size, block_size, 0, cuda_stream>>>(
			reinterpret_cast<typename mtk::cu_cutoff::detail::base_t<T>::type*>(ptr),
			m * detail::base_t<T>::num_elements, n,
			ld * detail::base_t<T>::num_elements,
			threshold
			);
}

#define CUTOFF_INSTANCE(x_type)\
	template void mtk::cu_cutoff::cutoff_small_abs_values<x_type>( \
		x_type* const ptr, \
		const std::uint32_t m, \
		const std::uint32_t n, \
		const std::uint32_t ld, \
		const typename mtk::cu_cutoff::detail::base_t<x_type>::type threshold, \
		cudaStream_t cuda_stream \
		)

CUTOFF_INSTANCE(half   );
CUTOFF_INSTANCE(float  );
CUTOFF_INSTANCE(double );
CUTOFF_INSTANCE(half2  );
CUTOFF_INSTANCE(float2 );
CUTOFF_INSTANCE(double2);
