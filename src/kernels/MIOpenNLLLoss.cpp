/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view.hpp"
#include "block_reduce.hpp"

#ifndef D_TYPE
#define D_TYPE float
#endif

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

template <typename T>
__device__ void nlllossForward5d(const T* __restrict__ input,
                                 const uint64_t* __restrict__ target,
                                 const T* __restrict__ weight,
                                 T* __restrict__ loss_sum,
                                 uint64_t ignore_index,
                                 float divisor,
                                 tensor_view_t<5> input_tv,
                                 tensor_view_t<4> target_tv,
                                 tensor_view_t<1> weight_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<4>(target_tv, gid);
    uint64_t n[4];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];
    n[3] = tensor_layout.layout[3];

    if(n[0] >= target_tv.size[0])
        return;

    uint64_t C = weight_tv.size[0];

    uint64_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
    uint64_t t            = target[target_index];

    uint64_t input_index = input_tv.get_tensor_view_idx({n[0], t, n[1], n[2], n[3]});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    if(t == ignore_index || t >= C)
    {
        loss_sum[gid] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_index]);
    FLOAT_ACCUM d           = !isnan(divisor) ? CVT_FP32_2ACCUM(divisor) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM val         = (-w * input_value) / d;
    loss_sum[gid]           = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossForward5d(const D_TYPE* __restrict__ input,
                                            const uint64_t* __restrict__ target,
                                            const D_TYPE* __restrict__ weight,
                                            D_TYPE* __restrict__ loss_sum,
                                            uint64_t ignore_index,
                                            float divisor,
                                            tensor_view_t<5> input_tv,
                                            tensor_view_t<4> target_tv,
                                            tensor_view_t<1> weight_tv)
{
    nlllossForward5d<D_TYPE>(
        input, target, weight, loss_sum, ignore_index, divisor, input_tv, target_tv, weight_tv);
}

template <typename T>
__device__ void nlllossBackward2d(T* __restrict__ input_grad,
                                  const uint64_t* __restrict__ target,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ output_grad,
                                  uint64_t ignore_index,
                                  float divisor,
                                  tensor_view_t<2> input_grad_tv,
                                  tensor_view_t<1> target_tv,
                                  tensor_view_t<1> weight_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= target_tv.size[0])
        return;

    uint64_t C            = weight_tv.size[0];
    uint64_t target_index = target_tv.get_tensor_view_idx({gid});
    uint64_t t            = target[target_index];

    uint64_t input_grad_index = input_grad_tv.get_tensor_view_idx({gid, t});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    if(t == ignore_index || t >= C)
    {
        input_grad[input_grad_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[0]);
    FLOAT_ACCUM d                = (divisor ? divisor : CVT_FP32_2ACCUM(1.0f));
    FLOAT_ACCUM input_grad_value = (-w * grad_val) / d;

    input_grad[input_grad_index] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossBackward2d(D_TYPE* __restrict__ input_grad,
                                             const uint64_t* __restrict__ target,
                                             const D_TYPE* __restrict__ weight,
                                             const D_TYPE* __restrict__ output_grad,
                                             uint64_t ignore_index,
                                             float divisor,
                                             tensor_view_t<2> input_grad_tv,
                                             tensor_view_t<1> target_tv,
                                             tensor_view_t<1> weight_tv)
{
    nlllossBackward2d<D_TYPE>(input_grad,
                              target,
                              weight,
                              output_grad,
                              ignore_index,
                              divisor,
                              input_grad_tv,
                              target_tv,
                              weight_tv);
}

template <typename T>
__device__ void nlllossBackward5d(T* __restrict__ input_grad,
                                  const uint64_t* __restrict__ target,
                                  const T* __restrict__ weight,
                                  const T* __restrict__ output_grad,
                                  uint64_t ignore_index,
                                  float divisor,
                                  tensor_view_t<5> input_grad_tv,
                                  tensor_view_t<4> target_tv,
                                  tensor_view_t<1> weight_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<4>(target_tv, gid);
    uint64_t n[4];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];
    n[3] = tensor_layout.layout[3];

    if(n[0] >= target_tv.size[0])
        return;

    uint64_t target_index = target_tv.get_tensor_view_idx(tensor_layout);

    uint64_t C                = weight_tv.size[0];
    uint64_t t                = target[target_index];
    uint64_t input_grad_index = input_grad_tv.get_tensor_view_idx({n[0], t, n[1], n[2], n[3]});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    if(t == ignore_index || t >= C)
    {
        input_grad[input_grad_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[0]);
    FLOAT_ACCUM d                = (divisor ? divisor : CVT_FP32_2ACCUM(1.0f));
    FLOAT_ACCUM input_grad_value = (-w * grad_val) / d;

    input_grad[input_grad_index] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossBackward5d(D_TYPE* __restrict__ input_grad,
                                             const uint64_t* __restrict__ target,
                                             const D_TYPE* __restrict__ weight,
                                             const D_TYPE* __restrict__ output_grad,
                                             uint64_t ignore_index,
                                             float divisor,
                                             tensor_view_t<5> input_grad_tv,
                                             tensor_view_t<4> target_tv,
                                             tensor_view_t<1> weight_tv)
{
    nlllossBackward5d<D_TYPE>(input_grad,
                              target,
                              weight,
                              output_grad,
                              ignore_index,
                              divisor,
                              input_grad_tv,
                              target_tv,
                              weight_tv);
}

template <typename T>
__device__ void nlllossUnreducedForward5d(const T* __restrict__ input,
                                          const uint64_t* __restrict__ target,
                                          const T* __restrict__ weight,
                                          T* __restrict__ output,
                                          uint64_t ignore_index,
                                          tensor_view_t<5> input_tv,
                                          tensor_view_t<4> target_tv,
                                          tensor_view_t<1> weight_tv,
                                          tensor_view_t<4> output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<4>(output_tv, gid);
    uint64_t n[4];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];
    n[3] = tensor_layout.layout[3];

    if(n[0] >= output_tv.size[0])
        return;

    uint64_t C = weight_tv.size[0];

    uint64_t target_index = target_tv.get_tensor_view_idx({n[0], n[1], n[2], n[3]});
    uint64_t t            = target[target_index];

    uint64_t input_index = input_tv.get_tensor_view_idx({n[0], t, n[1], n[2], n[3]});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    uint64_t output_index = output_tv.get_tensor_view_idx(tensor_layout);

    if(t == ignore_index || t >= C)
    {
        output[output_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_index]);

    FLOAT_ACCUM val      = -w * input_value;
    output[output_index] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward5d(const D_TYPE* __restrict__ input,
                                                     const uint64_t* __restrict__ target,
                                                     const D_TYPE* __restrict__ weight,
                                                     D_TYPE* __restrict__ output,
                                                     uint64_t ignore_index,
                                                     tensor_view_t<5> input_tv,
                                                     tensor_view_t<4> target_tv,
                                                     tensor_view_t<1> weight_tv,
                                                     tensor_view_t<4> output_tv)
{
    nlllossUnreducedForward5d<D_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename T>
__device__ void nlllossUnreducedForward4d(const T* __restrict__ input,
                                          const uint64_t* __restrict__ target,
                                          const T* __restrict__ weight,
                                          T* __restrict__ output,
                                          uint64_t ignore_index,
                                          tensor_view_t<4> input_tv,
                                          tensor_view_t<3> target_tv,
                                          tensor_view_t<1> weight_tv,
                                          tensor_view_t<3> output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<3>(output_tv, gid);
    uint64_t n[3];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];

    if(n[0] >= output_tv.size[0])
        return;

    uint64_t C = weight_tv.size[0];

    uint64_t target_index = target_tv.get_tensor_view_idx({n[0], n[1], n[2]});
    uint64_t t            = target[target_index];

    uint64_t input_index = input_tv.get_tensor_view_idx({n[0], t, n[1], n[2]});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    uint64_t output_index = output_tv.get_tensor_view_idx({n[0], n[1], n[2]});

    if(t == ignore_index || t >= C)
    {
        output[output_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_index]);

    FLOAT_ACCUM val      = -w * input_value;
    output[output_index] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward4d(const D_TYPE* __restrict__ input,
                                                     const uint64_t* __restrict__ target,
                                                     const D_TYPE* __restrict__ weight,
                                                     D_TYPE* __restrict__ output,
                                                     uint64_t ignore_index,
                                                     tensor_view_t<4> input_tv,
                                                     tensor_view_t<3> target_tv,
                                                     tensor_view_t<1> weight_tv,
                                                     tensor_view_t<3> output_tv)
{
    nlllossUnreducedForward4d<D_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename T>
__device__ void nlllossUnreducedForward4dContiguous(const T* __restrict__ input,
                                                    const uint64_t* __restrict__ target,
                                                    const T* __restrict__ weight,
                                                    T* __restrict__ output,
                                                    uint64_t ignore_index,
                                                    tensor_view_t<4> input_tv,
                                                    tensor_view_t<3> output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<3>(output_tv, gid);
    uint64_t n[3];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];

    if(n[0] >= output_tv.size[0])
        return;

    uint64_t C           = input_tv.size[1];
    uint64_t t           = target[gid];
    uint64_t input_index = input_tv.get_tensor_view_idx({n[0], t, n[1], n[2]});

    if(t == ignore_index || t >= C)
    {
        output[gid] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_index]);

    FLOAT_ACCUM val = -w * input_value;
    output[gid]     = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward4dContiguous(const D_TYPE* __restrict__ input,
                                                               const uint64_t* __restrict__ target,
                                                               const D_TYPE* __restrict__ weight,
                                                               D_TYPE* __restrict__ output,
                                                               uint64_t ignore_index,
                                                               tensor_view_t<4> input_tv,
                                                               tensor_view_t<3> output_tv)
{
    nlllossUnreducedForward4dContiguous<D_TYPE>(
        input, target, weight, output, ignore_index, input_tv, output_tv);
}

template <typename T>
__device__ void nlllossUnreducedForward2d(const T* __restrict__ input,
                                          const uint64_t* __restrict__ target,
                                          const T* __restrict__ weight,
                                          T* __restrict__ output,
                                          uint64_t ignore_index,
                                          tensor_view_t<2> input_tv,
                                          tensor_view_t<1> target_tv,
                                          tensor_view_t<1> weight_tv,
                                          tensor_view_t<1> output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= output_tv.size[0])
        return;

    uint64_t C = weight_tv.size[0];

    uint64_t target_index = target_tv.get_tensor_view_idx({gid});
    uint64_t t            = target[target_index];

    uint64_t input_index = input_tv.get_tensor_view_idx({gid, t});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    uint64_t output_index = output_tv.get_tensor_view_idx({gid});

    if(t == ignore_index || t >= C)
    {
        output[output_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);

    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_index]);

    FLOAT_ACCUM val      = -w * input_value;
    output[output_index] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward2d(const D_TYPE* __restrict__ input,
                                                     const uint64_t* __restrict__ target,
                                                     const D_TYPE* __restrict__ weight,
                                                     D_TYPE* __restrict__ output,
                                                     uint64_t ignore_index,
                                                     tensor_view_t<2> input_tv,
                                                     tensor_view_t<1> target_tv,
                                                     tensor_view_t<1> weight_tv,
                                                     tensor_view_t<1> output_tv)
{
    nlllossUnreducedForward2d<D_TYPE>(
        input, target, weight, output, ignore_index, input_tv, target_tv, weight_tv, output_tv);
}

template <typename T>
__device__ void nlllossUnreducedForward2dContiguous(const T* __restrict__ input,
                                                    const uint64_t* __restrict__ target,
                                                    const T* __restrict__ weight,
                                                    T* __restrict__ output,
                                                    uint64_t ignore_index,
                                                    tensor_view_t<1> weight_tv,
                                                    tensor_view_t<1> output_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= output_tv.size[0])
        return;

    uint64_t t = target[gid];
    uint64_t C = weight_tv.size[0];
    if(t == ignore_index || t >= C)
    {
        output[gid] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);

    uint64_t input_offset   = gid * C + t;
    FLOAT_ACCUM input_value = CVT_FLOAT2ACCUM(input[input_offset]);

    FLOAT_ACCUM val = -w * input_value;
    output[gid]     = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void NLLLossUnreducedForward2dContiguous(const D_TYPE* __restrict__ input,
                                                               const uint64_t* __restrict__ target,
                                                               const D_TYPE* __restrict__ weight,
                                                               D_TYPE* __restrict__ output,
                                                               uint64_t ignore_index,
                                                               tensor_view_t<1> weight_tv,
                                                               tensor_view_t<1> output_tv)
{
    nlllossUnreducedForward2dContiguous<D_TYPE>(
        input, target, weight, output, ignore_index, weight_tv, output_tv);
}

template <typename T>
__device__ void nlllossUnreducedBackward5d(T* __restrict__ input_grad,
                                           const uint64_t* __restrict__ target,
                                           const T* __restrict__ weight,
                                           const T* __restrict__ output_grad,
                                           uint64_t ignore_index,
                                           tensor_view_t<5> input_grad_tv,
                                           tensor_view_t<4> target_tv,
                                           tensor_view_t<1> weight_tv,
                                           tensor_view_t<4> output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<4>(output_grad_tv, gid);
    uint64_t n[4];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];
    n[3] = tensor_layout.layout[3];

    if(n[0] >= output_grad_tv.size[0])
        return;

    uint64_t C            = weight_tv.size[0];
    uint64_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
    uint64_t t            = target[target_index];

    uint64_t input_grad_index = input_grad_tv.get_tensor_view_idx({n[0], t, n[1], n[2], n[3]});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    uint64_t output_grad_index = output_grad_tv.get_tensor_view_idx(tensor_layout);

    if(t == ignore_index || t >= C)
    {
        input_grad[input_grad_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[output_grad_index]);
    FLOAT_ACCUM input_grad_value = -w * grad_val;

    input_grad[input_grad_index] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossUnreducedBackward5d(D_TYPE* __restrict__ input_grad,
                                                      const uint64_t* __restrict__ target,
                                                      const D_TYPE* __restrict__ weight,
                                                      const D_TYPE* __restrict__ output_grad,
                                                      uint64_t ignore_index,
                                                      tensor_view_t<5> input_grad_tv,
                                                      tensor_view_t<4> target_tv,
                                                      tensor_view_t<1> weight_tv,
                                                      tensor_view_t<4> output_grad_tv)
{
    nlllossUnreducedBackward5d<D_TYPE>(input_grad,
                                       target,
                                       weight,
                                       output_grad,
                                       ignore_index,
                                       input_grad_tv,
                                       target_tv,
                                       weight_tv,
                                       output_grad_tv);
}

template <typename T>
__device__ void nlllossUnreducedBackward4d(T* __restrict__ input_grad,
                                           const uint64_t* __restrict__ target,
                                           const T* __restrict__ weight,
                                           const T* __restrict__ output_grad,
                                           uint64_t ignore_index,
                                           tensor_view_t<4> input_grad_tv,
                                           tensor_view_t<3> target_tv,
                                           tensor_view_t<1> weight_tv,
                                           tensor_view_t<3> output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<3>(output_grad_tv, gid);
    uint64_t n[3];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];

    if(n[0] >= output_grad_tv.size[0])
        return;

    uint64_t C            = weight_tv.size[0];
    uint64_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
    uint64_t t            = target[target_index];

    uint64_t input_grad_index = input_grad_tv.get_tensor_view_idx({n[0], t, n[1], n[2]});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    uint64_t output_grad_index = output_grad_tv.get_tensor_view_idx(tensor_layout);

    if(t == ignore_index || t >= C)
    {
        input_grad[input_grad_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[output_grad_index]);
    FLOAT_ACCUM input_grad_value = -w * grad_val;

    input_grad[input_grad_index] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossUnreducedBackward4d(D_TYPE* __restrict__ input_grad,
                                                      const uint64_t* __restrict__ target,
                                                      const D_TYPE* __restrict__ weight,
                                                      const D_TYPE* __restrict__ output_grad,
                                                      uint64_t ignore_index,
                                                      tensor_view_t<4> input_grad_tv,
                                                      tensor_view_t<3> target_tv,
                                                      tensor_view_t<1> weight_tv,
                                                      tensor_view_t<3> output_grad_tv)
{
    nlllossUnreducedBackward4d<D_TYPE>(input_grad,
                                       target,
                                       weight,
                                       output_grad,
                                       ignore_index,
                                       input_grad_tv,
                                       target_tv,
                                       weight_tv,
                                       output_grad_tv);
}

template <typename T>
__device__ void nlllossUnreducedBackward4dContiguous(T* __restrict__ input_grad,
                                                     const uint64_t* __restrict__ target,
                                                     const T* __restrict__ weight,
                                                     const T* __restrict__ output_grad,
                                                     uint64_t ignore_index,
                                                     tensor_view_t<4> input_grad_tv,
                                                     tensor_view_t<1> weight_tv,
                                                     tensor_view_t<3> output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    auto tensor_layout = tensor_layout_t<3>(output_grad_tv, gid);
    uint64_t n[3];
    n[0] = tensor_layout.layout[0];
    n[1] = tensor_layout.layout[1];
    n[2] = tensor_layout.layout[2];

    if(n[0] >= output_grad_tv.size[0])
        return;

    uint64_t C = weight_tv.size[0];
    uint64_t t = target[gid];

    uint64_t input_grad_index = input_grad_tv.get_tensor_view_idx({n[0], t, n[1], n[2]});

    if(t == ignore_index || t >= C)
    {
        input_grad[input_grad_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w        = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val = CVT_FLOAT2ACCUM(output_grad[gid]);
    FLOAT_ACCUM input_grad_value = -w * grad_val;

    input_grad[input_grad_index] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void
NLLLossUnreducedBackward4dContiguous(D_TYPE* __restrict__ input_grad,
                                     const uint64_t* __restrict__ target,
                                     const D_TYPE* __restrict__ weight,
                                     const D_TYPE* __restrict__ output_grad,
                                     uint64_t ignore_index,
                                     tensor_view_t<4> input_grad_tv,
                                     tensor_view_t<1> weight_tv,
                                     tensor_view_t<3> output_grad_tv)
{
    nlllossUnreducedBackward4dContiguous<D_TYPE>(input_grad,
                                                 target,
                                                 weight,
                                                 output_grad,
                                                 ignore_index,
                                                 input_grad_tv,
                                                 weight_tv,
                                                 output_grad_tv);
}

template <typename T>
__device__ void nlllossUnreducedBackward2d(T* __restrict__ input_grad,
                                           const uint64_t* __restrict__ target,
                                           const T* __restrict__ weight,
                                           const T* __restrict__ output_grad,
                                           uint64_t ignore_index,
                                           tensor_view_t<2> input_grad_tv,
                                           tensor_view_t<1> target_tv,
                                           tensor_view_t<1> weight_tv,
                                           tensor_view_t<1> output_grad_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= target_tv.size[0])
        return;

    uint64_t C            = weight_tv.size[0];
    auto tensor_layout    = tensor_layout_t<1>(target_tv, gid);
    uint64_t target_index = target_tv.get_tensor_view_idx(tensor_layout);
    uint64_t t            = target[target_index];

    uint64_t input_grad_index = input_grad_tv.get_tensor_view_idx({gid, t});

    uint64_t weight_index = weight_tv.get_tensor_view_idx({t});

    uint64_t output_grad_index = output_grad_tv.get_tensor_view_idx(tensor_layout);

    if(t == ignore_index || t >= C)
    {
        input_grad[input_grad_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w =
        weight != nullptr ? CVT_FLOAT2ACCUM(weight[weight_index]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val         = CVT_FLOAT2ACCUM(output_grad[output_grad_index]);
    FLOAT_ACCUM input_grad_value = -w * grad_val;

    input_grad[input_grad_index] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void NLLLossUnreducedBackward2d(D_TYPE* __restrict__ input_grad,
                                                      const uint64_t* __restrict__ target,
                                                      const D_TYPE* __restrict__ weight,
                                                      const D_TYPE* __restrict__ output_grad,
                                                      uint64_t ignore_index,
                                                      tensor_view_t<2> input_grad_tv,
                                                      tensor_view_t<1> target_tv,
                                                      tensor_view_t<1> weight_tv,
                                                      tensor_view_t<1> output_grad_tv)
{
    nlllossUnreducedBackward2d<D_TYPE>(input_grad,
                                       target,
                                       weight,
                                       output_grad,
                                       ignore_index,
                                       input_grad_tv,
                                       target_tv,
                                       weight_tv,
                                       output_grad_tv);
}

template <typename T>
__device__ void nlllossUnreducedBackward2dContiguous(T* __restrict__ input_grad,
                                                     const uint64_t* __restrict__ target,
                                                     const T* __restrict__ weight,
                                                     const T* __restrict__ output_grad,
                                                     uint64_t ignore_index,
                                                     tensor_view_t<2> input_grad_tv,
                                                     tensor_view_t<1> target_tv,
                                                     tensor_view_t<1> weight_tv)
{
    uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    if(gid >= target_tv.size[0])
        return;

    uint64_t C = weight_tv.size[0];
    uint64_t t = target[gid];

    uint64_t input_grad_index = input_grad_tv.get_tensor_view_idx({gid, t});

    if(t == ignore_index || t >= C)
    {
        input_grad[input_grad_index] = static_cast<T>(0);
        return;
    }

    FLOAT_ACCUM w        = weight != nullptr ? CVT_FLOAT2ACCUM(weight[t]) : CVT_FP32_2ACCUM(1.0f);
    FLOAT_ACCUM grad_val = CVT_FLOAT2ACCUM(output_grad[gid]);
    FLOAT_ACCUM input_grad_value = -w * grad_val;

    input_grad[input_grad_index] = CVT_ACCUM2FLOAT(input_grad_value);
}

extern "C" __global__ void
NLLLossUnreducedBackward2dContiguous(D_TYPE* __restrict__ input_grad,
                                     const uint64_t* __restrict__ target,
                                     const D_TYPE* __restrict__ weight,
                                     const D_TYPE* __restrict__ output_grad,
                                     uint64_t ignore_index,
                                     tensor_view_t<2> input_grad_tv,
                                     tensor_view_t<1> target_tv,
                                     tensor_view_t<1> weight_tv)
{
    nlllossUnreducedBackward2dContiguous<D_TYPE>(
        input_grad, target, weight, output_grad, ignore_index, input_grad_tv, target_tv, weight_tv);
}

template <typename DTYPE>
__device__ void lossSum(const DTYPE* input, DTYPE* output, size_t N)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? CVT_FLOAT2ACCUM(input[gid]) : CVT_FP32_2ACCUM(0.0f);
    val             = block_reduce<BinaryOp_t::Add, REDUCE_SIZE, ReduceThreadDim::X>(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void
LossSum(const D_TYPE* __restrict__ input, D_TYPE* __restrict__ output, size_t N)
{
    lossSum<D_TYPE>(input, output, N);
}
