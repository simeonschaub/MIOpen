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

template <typename TIO>
__device__ void L1LossReducedForward5d_kernel(const TIO* I,
                                              const TIO* T,
                                              FLOAT_ACCUM* lsum,
                                              const size_t divisor,
                                              tensor_view_t<5> I_tv,
                                              tensor_view_t<5> T_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const float div  = static_cast<float>(divisor);
    tensor_layout_t<5> input_layout(I_tv, gid);

    if(input_layout.layout[0] >= I_tv.size[0])
        return;

    size_t Iidx = I_tv.get_tensor_view_idx(input_layout);
    size_t Tidx = T_tv.get_tensor_view_idx(input_layout);

    FLOAT_ACCUM diff = abs(CVT_FLOAT2ACCUM(I[Iidx]) - CVT_FLOAT2ACCUM(T[Tidx])) / div;
    lsum[gid]        = diff;
}

extern "C" __global__ void L1LossReducedForward5d(const IO_TYPE* I,
                                                  const IO_TYPE* T,
                                                  FLOAT_ACCUM* lsum,
                                                  const size_t divisor,
                                                  tensor_view_t<5> I_tv,
                                                  tensor_view_t<5> T_tv)
{
    L1LossReducedForward5d_kernel<IO_TYPE>(I, T, lsum, divisor, I_tv, T_tv);
}
