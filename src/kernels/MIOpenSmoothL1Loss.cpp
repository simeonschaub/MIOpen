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
#include "tensor_view_5d.hpp"

#ifndef INPUT_TYPE
#define INPUT_TYPE float
#endif

#ifndef TARGET_TYPE
#define TARGET_TYPE float
#endif

#ifndef OUTPUT_TYPE
#define OUTPUT_TYPE float
#endif

#ifndef D_TYPE
#define D_TYPE float
#endif

#ifndef REDUCE_SIZE
#define REDUCE_SIZE 256
#endif

template <typename TI, typename TT, typename TO>
__device__ void smoothl1lossunreducedforwardcontiguous(
    const TI* I, const TT* T, TO* O, const float beta, const ulong n)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= n)
        return;

    FLOAT_ACCUM diff = fabs(CVT_FLOAT2ACCUM(I[gid]) - CVT_FLOAT2ACCUM(T[gid]));
    O[gid] = CVT_ACCUM2FLOAT(diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta);
}

extern "C" __global__ void SmoothL1LossUnreducedForwardContiguous(const INPUT_TYPE* __restrict__ I,
                                                                  const TARGET_TYPE* __restrict__ T,
                                                                  OUTPUT_TYPE* __restrict__ O,
                                                                  const float beta,
                                                                  const ulong n)
{
    // instantiate the kernel
    smoothl1lossunreducedforwardcontiguous<INPUT_TYPE, TARGET_TYPE, OUTPUT_TYPE>(I, T, O, beta, n);
}

template <typename TI, typename TT, typename TO>
__device__ void smoothl1lossunreducedforward5d(const TI* I,
                                               const TT* T,
                                               TO* O,
                                               const float beta,
                                               tensor_view_5d_t I_tv,
                                               tensor_view_5d_t T_tv,
                                               tensor_view_5d_t O_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, O_tv);

    if(n[0] >= O_tv.size[0])
        return;

    size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
    size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
    size_t Oidx = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

    FLOAT_ACCUM diff = fabs(CVT_FLOAT2ACCUM(I[Iidx]) - CVT_FLOAT2ACCUM(T[Tidx]));
    O[Oidx]          = CVT_ACCUM2FLOAT(diff < beta ? 0.5 * diff * diff / beta : diff - 0.5 * beta);
}

extern "C" __global__ void SmoothL1LossUnreducedForward5d(const INPUT_TYPE* __restrict__ I,
                                                          const TARGET_TYPE* __restrict__ T,
                                                          OUTPUT_TYPE* __restrict__ O,
                                                          const float beta,
                                                          tensor_view_5d_t I_tv,
                                                          tensor_view_5d_t T_tv,
                                                          tensor_view_5d_t O_tv)
{
    // instantiate the kernel
    smoothl1lossunreducedforward5d<INPUT_TYPE, TARGET_TYPE, OUTPUT_TYPE>(
        I, T, O, beta, I_tv, T_tv, O_tv);
}

template <typename TI, typename TT, typename TO>
__device__ void smoothl1lossreducedforward5d(const TI* I,
                                             const TT* T,
                                             TO* lsum,
                                             const float beta,
                                             const float divisor,
                                             tensor_view_5d_t I_tv,
                                             tensor_view_5d_t T_tv)
{
    const size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n[5];
    GET_NCDHW(n[0], n[1], n[2], n[3], n[4], gid, I_tv);

    if(n[0] >= I_tv.size[0])
        return;

    size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
    size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

    float diff = abs(CVT_FLOAT2ACCUM(I[Iidx]) - CVT_FLOAT2ACCUM(T[Tidx]));
    lsum[Iidx] =
        CVT_ACCUM2FLOAT((diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta) / divisor);
}

extern "C" __global__ void SmoothL1LossReducedForward5d(const INPUT_TYPE* __restrict__ I,
                                                        const TARGET_TYPE* __restrict__ T,
                                                        OUTPUT_TYPE* __restrict__ lsum,
                                                        const float beta,
                                                        const float divisor,
                                                        tensor_view_5d_t I_tv,
                                                        tensor_view_5d_t T_tv)
{
    // instantiate the kernel
    smoothl1lossreducedforward5d<INPUT_TYPE, TARGET_TYPE, OUTPUT_TYPE>(
        I, T, lsum, beta, divisor, I_tv, T_tv);
}

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

template <typename DTYPE>
__device__ void losssum(const DTYPE* input, DTYPE* output, size_t N)
{
    auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    FLOAT_ACCUM val = gid < N ? CVT_FLOAT2ACCUM(input[gid]) : static_cast<FLOAT_ACCUM>(0.0f);
    val             = block_reduce_sum(val);

    if(threadIdx.x == 0)
        output[blockIdx.x] = CVT_ACCUM2FLOAT(val);
}

extern "C" __global__ void
LossSum(const D_TYPE* __restrict__ input, D_TYPE* __restrict__ output, size_t N)
{
    // instantiate the kernel
    losssum<D_TYPE>(input, output, N);
}
