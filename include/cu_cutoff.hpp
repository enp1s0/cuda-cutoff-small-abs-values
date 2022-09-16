#pragma once
#include <cstdint>
#include <cuda_fp16.h>

namespace mtk {
namespace cu_cutoff {
namespace detail {
template <class T>
struct base_t {
	using type = T;
	static const unsigned num_elements = 1;
};
template <> struct base_t<double2> {using type = double;static const unsigned num_elements = 2;};
template <> struct base_t<float2 > {using type = float; static const unsigned num_elements = 2;};
template <> struct base_t<half2  > {using type = half;  static const unsigned num_elements = 2;};
} // namespace detail
template <class T>
void cutoff_small_abs_values(
		T* const ptr,
		const std::uint32_t m,
		const std::uint32_t n,
		const std::uint32_t ld,
		const typename detail::base_t<T>::type threshold,
		cudaStream_t cuda_stream = 0
		);
} // namespace cu_cutoff
} // namespace mtk
