#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using cat_serial_fn = void(*)(Tensor &, TensorList, int64_t);
using channel_last_cat_channel_last_tensor_fn = void(*)(Tensor &, TensorList);
DECLARE_DISPATCH(cat_serial_fn, cat_serial_stub);
DECLARE_DISPATCH(
    channel_last_cat_channel_last_tensor_fn,
    channel_dim_cat_channels_last_stub);

}}  // namespace at::native
