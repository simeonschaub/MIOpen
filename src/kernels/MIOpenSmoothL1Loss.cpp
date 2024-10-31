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
#include "MIOpenLossReductionMode.hpp"

template <typename TIO, uint32_t NDIM, LossReductionMode_t REDUCTION_T>
__device__ void SmoothL1LossForward(const TIO* I,
                                    const TIO* T,
                                    void* O,
                                    const float beta,
                                    const uint64_t size,
                                    tensor_view_t<NDIM> I_tv,
                                    tensor_view_t<NDIM> T_tv,
                                    tensor_view_t<NDIM> O_tv)
{
    const uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    tensor_layout_t<5> tensor_layout(I_tv, gid);
    if(tensor_layout.layout[0] >= I_tv.size[0])
        return;

    FLOAT_ACCUM i    = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM t    = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM diff = abs(i - t);
    FLOAT_ACCUM loss = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;

    switch(REDUCTION_T)
    {
    case LossReductionMode_t::NONE:
        static_cast<TIO*>(O)[O_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(loss);
        break;
    case LossReductionMode_t::SUM: static_cast<FLOAT_ACCUM*>(O)[gid] = loss; break;
    case LossReductionMode_t::MEAN: static_cast<FLOAT_ACCUM*>(O)[gid] = loss / size; break;
    default: break;
    }
}

extern "C" __global__ void SmoothL1LossForward(const FLOAT* __restrict__ I,
                                               const FLOAT* __restrict__ T,
                                               void* __restrict__ O,
                                               const float beta,
                                               const uint64_t size,
                                               tensor_view_t<VIEW_DIMS> I_tv,
                                               tensor_view_t<VIEW_DIMS> T_tv,
                                               tensor_view_t<VIEW_DIMS> O_tv)
{
    // instantiate the kernel
    SmoothL1LossForward<FLOAT, VIEW_DIMS, static_cast<LossReductionMode_t>(REDUCTION_TYPE)>(
        I, T, O, beta, size, I_tv, T_tv, O_tv);
}

template <typename TIO, uint32_t NDIM, LossReductionMode_t REDUCTION_T>
__device__ void SmoothL1LossBackward(const TIO* I,
                                     const TIO* T,
                                     const TIO* dO,
                                     TIO* dI,
                                     TIO* dT,
                                     float beta,
                                     const uint64_t size,
                                     tensor_view_t<NDIM> I_tv,
                                     tensor_view_t<NDIM> T_tv,
                                     tensor_view_t<NDIM> dO_tv,
                                     tensor_view_t<NDIM> dI_tv,
                                     tensor_view_t<NDIM> dT_tv)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    tensor_layout_t<5> tensor_layout(I_tv, gid);
    if(tensor_layout.layout[0] >= I_tv.size[0])
        return;

    FLOAT_ACCUM i = CVT_FLOAT2ACCUM(I[I_tv.get_tensor_view_idx(tensor_layout)]);
    FLOAT_ACCUM t = CVT_FLOAT2ACCUM(T[T_tv.get_tensor_view_idx(tensor_layout)]);

    FLOAT_ACCUM sub = i - t;
    FLOAT_ACCUM grad;

    switch(REDUCTION_T)
    {
    case LossReductionMode_t::MEAN:
        if(fabs(sub) < beta)
            grad = sub / beta * CVT_FLOAT2ACCUM(dO[0]) / size;
        else
            grad = (sub >= 0 ? 1.0f : -1.0f) * CVT_FLOAT2ACCUM(dO[0]) / size;
        break;
    default:
        if(fabs(sub) < beta)
            grad = sub / beta * CVT_FLOAT2ACCUM(dO[0]);
        else
            grad = (sub >= 0 ? 1.0f : -1.0f) * CVT_FLOAT2ACCUM(dO[0]);
        break;
    }

    if(dI)
        dI[dI_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(grad);
    if(dT)
        dT[dI_tv.get_tensor_view_idx(tensor_layout)] = CVT_ACCUM2FLOAT(-grad);
}

extern "C" __global__ void SmoothL1LossBackward(const FLOAT* __restrict__ I,
                                                const FLOAT* __restrict__ T,
                                                const FLOAT* __restrict__ dO,
                                                FLOAT* __restrict__ dI,
                                                FLOAT* __restrict__ dT,
                                                float beta,
                                                const uint64_t size,
                                                tensor_view_t<VIEW_DIMS> I_tv,
                                                tensor_view_t<VIEW_DIMS> T_tv,
                                                tensor_view_t<VIEW_DIMS> dO_tv,
                                                tensor_view_t<VIEW_DIMS> dI_tv,
                                                tensor_view_t<VIEW_DIMS> dT_tv)
{
    SmoothL1LossBackward<FLOAT, VIEW_DIMS, static_cast<LossReductionMode_t>(REDUCTION_TYPE)>(
        I, T, dO, dI, dT, beta, size, I_tv, T_tv, dO_tv, dI_tv, dT_tv);
}
