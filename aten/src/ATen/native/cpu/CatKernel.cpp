#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/Parallel.h>

namespace at { namespace native {

namespace {

struct InputMeta {
  void* data_ptr;
  int64_t inner_size;

  InputMeta(const Tensor& t, int64_t dim, int64_t inner)
    : data_ptr(t.data_ptr())
    , inner_size(t.size(dim) * inner) {}
};

template <typename scalar_t>
void cat_serial_kernel_impl(Tensor& result, TensorList tensors, int64_t dim) {
  auto size = result.sizes().vec();
  int64_t outer = 1, inner = 1;
  for (int64_t i = 0; i < dim; i++) {
    outer *= size[i];
  }
  for (int64_t i = dim + 1; i < size.size(); i++) {
    inner *= size[i];
  }
  scalar_t* result_data = result.data_ptr<scalar_t>();
  int64_t ninputs = tensors.size();
  std::vector<InputMeta> inputs;
  inputs.reserve(ninputs);
  for (auto const &tensor : tensors) {
    inputs.emplace_back(tensor, dim, inner);
  }

  using Vec = vec256::Vec256<scalar_t>;
  int64_t offset = 0;
  for (int64_t i = 0; i < outer; i++) {
    for (int64_t j = 0; j < ninputs; j++) {
      scalar_t* result_ptr = result_data + offset;
      int64_t local_inner = inputs[j].inner_size;
      scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * local_inner;
      if (local_inner < Vec::size()) {
        #ifndef _MSC_VER
        # pragma unroll
        #endif
        for (int64_t k = 0; k < local_inner; k++) {
          result_ptr[k] = input_ptr[k];
        }
      } else {
        vec256::map(
            [](Vec x) { return x; },
            result_ptr,
            input_ptr,
            local_inner);
      }
      offset += local_inner;
    }
  }
}

void cat_serial_kernel(Tensor& result, TensorList tensors, int64_t dim) {
  AT_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cat_serial_kernel", [&]() {
    cat_serial_kernel_impl<scalar_t>(result, tensors, dim);
  });
}

template <typename scalar_t>
void channel_dim_cat_kernel_channels_last_impl(
    Tensor& result, TensorList tensors) {
  int64_t outer_dims_size{1};
  // Assumes all the tensors are of the same size.
  // Only exception is channels dim.
  TORCH_CHECK((tensors[0].ndimension() == 4) || (tensors[0].ndimension() == 5),
      "Tensor must be either 4 or 5 dims to utilize this kernel.");
  if (tensors[0].ndimension() == 4) {
    outer_dims_size = outer_dims_size * tensors[0].size(0); // batch
    outer_dims_size = outer_dims_size * tensors[0].size(2); // H
    outer_dims_size = outer_dims_size * tensors[0].size(3); // W
  } else {
    outer_dims_size = outer_dims_size * tensors[0].size(0); // batch
    outer_dims_size = outer_dims_size * tensors[0].size(2); // D
    outer_dims_size = outer_dims_size * tensors[0].size(3); // H
    outer_dims_size = outer_dims_size * tensors[0].size(4); // W
  }
  int64_t output_stride = result.size(1);

  auto loop_nd = [&](int64_t start, int64_t end) {
    for (auto i = start; i < end; ++i) {
      scalar_t* output_ptr = result.data_ptr<scalar_t>();
      output_ptr += i * output_stride;
      for (const auto& t : tensors) {
        scalar_t* src_ptr = t.data_ptr<scalar_t>() + i * t.size(1);
        std::memcpy(output_ptr, src_ptr, t.size(1) * sizeof(scalar_t));
        output_ptr += t.size(1);
      }
    }
  };

  at::parallel_for(0, outer_dims_size, at::internal::GRAIN_SIZE, loop_nd);
}

void channels_dim_cat_channels_last_kernel(Tensor& result, TensorList tensors) {
  AT_DISPATCH_FLOATING_TYPES(
      result.scalar_type(), "channels_dim_cat_channels_last_kernel", [&]() {
    channel_dim_cat_kernel_channels_last_impl<scalar_t>(result, tensors);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel);
REGISTER_DISPATCH(
    channel_dim_cat_channels_last_stub, &channels_dim_cat_channels_last_kernel);

}} // at::native
