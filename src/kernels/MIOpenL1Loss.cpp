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
#include <cstddef>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"
#include "tensor_view_5d.hpp"

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

__device__ FLOAT_ACCUM warp_reduce_sum(FLOAT_ACCUM val)
{
    if(warpSize >= 64)
        val += __shfl_down(val, 32);
    if(warpSize >= 32)
        val += __shfl_down(val, 16);
    if(warpSize >= 16)
        val += __shfl_down(val, 8);
    if(warpSize >= 8)
        val += __shfl_down(val, 4);
    if(warpSize >= 4)
        val += __shfl_down(val, 2);
    if(warpSize >= 2)
        val += __shfl_down(val, 1);
    return val;
}

__device__ FLOAT_ACCUM block_reduce_sum(FLOAT_ACCUM val)
{
    static __shared__ FLOAT_ACCUM shared[REDUCE_SIZE / warpSize];
    auto lane = threadIdx.x % warpSize;
    auto wid  = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = threadIdx.x < REDUCE_SIZE / warpSize ? shared[lane] : 0;
    if(wid == 0)
        val = warp_reduce_sum(val);

    return val;
}

extern "C" __global__ void LossSum(const OUTPUT_TYPE* input, OUTPUT_TYPE* output, size_t N)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? CVT_FLOAT2ACCUM(input[gid]) : static_cast<FLOAT_ACCUM>(0.0f);
    val             = block_reduce_sum(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = CVT_ACCUM2FLOAT(val);
}

template <typename TI, typename TO>
__device__ void L1LossReducedForward5d_kernel(const TI* I,
                                              const TI* T,
                                              TO* lsum,
                                              const size_t divisor,
                                              tensor_view_5d_t I_tv,
                                              tensor_view_5d_t T_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n[5];
    const float div = static_cast<float>(divisor);
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, I_tv);

    if(n[0] >= I_tv.size[0])
        return;

    size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
    size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

    FLOAT_ACCUM diff = abs(CVT_FLOAT2ACCUM(I[Iidx]) - CVT_FLOAT2ACCUM(T[Tidx]));
    lsum[gid]        = CVT_ACCUM2FLOAT(diff / div);
}

extern "C" __global__ void L1LossReducedForward5d(const INPUT_TYPE* I,
                                                  const INPUT_TYPE* T,
                                                  OUTPUT_TYPE* lsum,
                                                  const size_t divisor,
                                                  tensor_view_5d_t I_tv,
                                                  tensor_view_5d_t T_tv)
{
    L1LossReducedForward5d_kernel<INPUT_TYPE, OUTPUT_TYPE>(I, T, lsum, divisor, I_tv, T_tv);
}

template <typename TI, typename TO>
__device__ void L1LossReducedBackward5d_kernel(const TI* I,
                                               const TI* T,
                                               const TI* dO,
                                               TO* dI,
                                               TO* dT,
                                               size_t divisor,
                                               tensor_view_5d_t I_tv,
                                               tensor_view_5d_t T_tv,
                                               tensor_view_5d_t dI_tv,
                                               tensor_view_5d_t dT_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, I_tv);

    if(n[0] >= I_tv.size[0])
        return;

    size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
    size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

    FLOAT_ACCUM Ival = CVT_FLOAT2ACCUM(I[Iidx]);
    FLOAT_ACCUM Tval = CVT_FLOAT2ACCUM(T[Tidx]);
    FLOAT_ACCUM grad =
        (Ival >= Tval) ? CVT_FLOAT2ACCUM(dO[0]) / divisor : -(CVT_FLOAT2ACCUM(dO[0]) / divisor);

    if(dI)
        dI[TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4])] = CVT_ACCUM2FLOAT(grad);
    if(dT)
        dT[TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4])] = CVT_ACCUM2FLOAT(-grad);
}

extern "C" __global__ void L1LossReducedBackward5d(const INPUT_TYPE* I,
                                                   const INPUT_TYPE* T,
                                                   const INPUT_TYPE* dO,
                                                   OUTPUT_TYPE* dI,
                                                   OUTPUT_TYPE* dT,
                                                   size_t divisor,
                                                   tensor_view_5d_t I_tv,
                                                   tensor_view_5d_t T_tv,
                                                   tensor_view_5d_t dI_tv,
                                                   tensor_view_5d_t dT_tv)
{
    L1LossReducedBackward5d_kernel<INPUT_TYPE, OUTPUT_TYPE>(
        I, T, dO, dI, dT, divisor, I_tv, T_tv, dI_tv, dT_tv);
}
