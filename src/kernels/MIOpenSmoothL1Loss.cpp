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

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE float
#endif

template <typename TI, typename TO>
__device__ void smoothl1lossunreducedforwardcontiguous(
    const TI* I, const TI* T, TO* O, const float beta, const ulong n)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= n)
        return;

    FLOAT_ACCUM diff = fabs(CVT_FLOAT2ACCUM(I[gid]) - CVT_FLOAT2ACCUM(T[gid]));
    O[gid] = CVT_ACCUM2FLOAT(diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta);
}

extern "C" __global__ void SmoothL1LossUnreducedForwardContiguous(const INPUT_TYPE* __restrict__ I,
                                                                  const INPUT_TYPE* __restrict__ T,
                                                                  OUTPUT_TYPE* __restrict__ O,
                                                                  const float beta,
                                                                  const ulong n)
{
    // instantiate the kernel
    smoothl1lossunreducedforwardcontiguous<INPUT_TYPE, OUTPUT_TYPE>(I, T, O, beta, n);
}
