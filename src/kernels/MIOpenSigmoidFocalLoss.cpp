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
__device__ void sigmoidFocalLossFwd(const TIO* __restrict__ input,
                                    const TIO* __restrict__ target,
                                    void* __restrict__ output,
                                    const float alpha,
                                    const float gamma,
                                    const uint64_t size,
                                    tensor_view_t<NDIM> input_tv,
                                    tensor_view_t<NDIM> target_tv,
                                    tensor_view_t<NDIM> output_tv)
{
    /*
        Dim: input = target = workspace = {N, C, D, H, W}.
        Each thread handle an elem in the input, target tensor.
        Lws = {LOCAL_SIZE_SIGMOIDFOCALLOSS(default = 256), 1, 1}.
        Gws = {AlignUp(N * C * D * H * W, lws.x), 1, 1}.
    */
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<NDIM> idx(input_tv, gid);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM i = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t = CVT_FLOAT2ACCUM(target[target_tv.get_tensor_view_idx(idx)]);

    /* The formula follows torchvision package: torchvision/ops/focal_loss.py */
    FLOAT_ACCUM p      = 1 / (1 + exp(-i));
    FLOAT_ACCUM ceLoss = -(t * log(p) + (1 - t) * log(1 - p));
    FLOAT_ACCUM pT     = p * t + (1 - p) * (1 - t);
    FLOAT_ACCUM loss   = ceLoss * pow(1 - pT, gamma);

    if(alpha >= 0)
    {
        FLOAT_ACCUM alpha_t = alpha * t + (1 - alpha) * (1 - t);
        loss                = alpha_t * loss;
    }

    switch(REDUCTION_T)
    {
    case LossReductionMode_t::NONE:
        static_cast<TIO*>(output)[output_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(loss);
        break;
    case LossReductionMode_t::SUM: static_cast<FLOAT_ACCUM*>(output)[gid] = loss; break;
    case LossReductionMode_t::MEAN: static_cast<FLOAT_ACCUM*>(output)[gid] = loss / size; break;
    default: break;
    }
}

extern "C" __global__ void SigmoidFocalLossFwd(const IN_OUT_TYPE* __restrict__ input,
                                               const IN_OUT_TYPE* __restrict__ target,
                                               void* __restrict__ output,
                                               const float alpha,
                                               const float gamma,
                                               const uint64_t size,
                                               tensor_view_t<VIEW_DIMS> input_tv,
                                               tensor_view_t<VIEW_DIMS> target_tv,
                                               tensor_view_t<VIEW_DIMS> output_tv)
{
    sigmoidFocalLossFwd<IN_OUT_TYPE, VIEW_DIMS, static_cast<LossReductionMode_t>(REDUCTION_TYPE)>(
        input, target, output, alpha, gamma, size, input_tv, target_tv, output_tv);
}

template <typename TIO, uint32_t NDIM, LossReductionMode_t REDUCTION_T>
__device__ void sigmoidFocalLossBwd(const TIO* __restrict__ input,
                                    const TIO* __restrict__ target,
                                    const TIO* __restrict__ doutput,
                                    TIO* __restrict__ dinput,
                                    TIO* __restrict__ dtarget,
                                    const float alpha,
                                    const float gamma,
                                    const uint64_t size,
                                    tensor_view_t<NDIM> input_tv,
                                    tensor_view_t<NDIM> target_tv,
                                    tensor_view_t<NDIM> doutput_tv,
                                    tensor_view_t<NDIM> dinput_tv,
                                    tensor_view_t<NDIM> dtarget_tv)
{
    /*
        Dim: input = target = doutput = dinput = dtarget = {N, C, D, H, W}.
        Each thread handle an elem in the input, target, doutput tensor.
        Lws = {LOCAL_SIZE_SIGMOIDFOCALLOSS(default = 256), 1, 1}.
        Gws = {AlignUp(N * C * D * H * W, lws.x), 1, 1}.
    */
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;

    tensor_layout_t<NDIM> idx(input_tv, gid);
    tensor_layout_t<NDIM> doIdx(doutput_tv, 0);
    if(idx.layout[0] >= input_tv.size[0])
        return;

    FLOAT_ACCUM i = CVT_FLOAT2ACCUM(input[input_tv.get_tensor_view_idx(idx)]);
    FLOAT_ACCUM t = CVT_FLOAT2ACCUM(target[target_tv.get_tensor_view_idx(idx)]);

    FLOAT_ACCUM dO;
    switch(REDUCTION_T)
    {
    case LossReductionMode_t::NONE:
        dO = CVT_FLOAT2ACCUM(doutput[doutput_tv.get_tensor_view_idx(idx)]);
        break;
    case LossReductionMode_t::SUM: dO = CVT_FLOAT2ACCUM(doutput[0]); break;
    case LossReductionMode_t::MEAN: dO = CVT_FLOAT2ACCUM(doutput[0]) / size; break;
    default: break;
    }

    /* Formula is formed by compute fwd's formula gradient */
    FLOAT_ACCUM p       = 1 / (1 + exp(-i));
    FLOAT_ACCUM ceLoss  = -(t * log(p) + (1 - t) * log(1 - p));
    FLOAT_ACCUM pT      = p * t + (1 - p) * (1 - t);
    FLOAT_ACCUM powPt   = pow(1 - pT, gamma);
    FLOAT_ACCUM alpha_t = alpha * t + (1 - alpha) * (1 - t);

    if(dinput)
    {
        FLOAT_ACCUM dpdi = exp(-i) / pow(1 + exp(-i), 2);
        // dceloss/di = dceloss/dp * dp/di
        FLOAT_ACCUM dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
        // dpowt/di = dpowt/dpT * dpT/dp * dp/di
        FLOAT_ACCUM dpowptdi = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

        // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
        FLOAT_ACCUM dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
        FLOAT_ACCUM grad = dO * dLdi;

        if(alpha >= 0)
            grad *= alpha_t;
        dinput[dinput_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(grad);
    }

    if(dtarget)
    {
        FLOAT_ACCUM dcelossdt = -log(p) + log(1 - p);
        FLOAT_ACCUM dpowptdt  = gamma * pow(1 - pT, gamma - 1) * (1 - 2 * p);
        // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
        FLOAT_ACCUM dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
        FLOAT_ACCUM gradTarget = dO * dLdt;

        if(alpha >= 0)
        {
            // alpha_t * dL/dt + dalpha_t/dt * L
            gradTarget = dO * (alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt);
        }
        dtarget[dtarget_tv.get_tensor_view_idx(idx)] = CVT_ACCUM2FLOAT(gradTarget);
    }
}

extern "C" __global__ void SigmoidFocalLossBwd(const IN_OUT_TYPE* __restrict__ input,
                                               IN_OUT_TYPE* __restrict__ target,
                                               IN_OUT_TYPE* __restrict__ doutput,
                                               IN_OUT_TYPE* __restrict__ dinput,
                                               IN_OUT_TYPE* __restrict__ dtarget,
                                               const float alpha,
                                               const float gamma,
                                               const uint64_t size,
                                               tensor_view_t<VIEW_DIMS> input_tv,
                                               tensor_view_t<VIEW_DIMS> target_tv,
                                               tensor_view_t<VIEW_DIMS> doutput_tv,
                                               tensor_view_t<VIEW_DIMS> dinput_tv,
                                               tensor_view_t<VIEW_DIMS> dtarget_tv)
{
    sigmoidFocalLossBwd<IN_OUT_TYPE, VIEW_DIMS, static_cast<LossReductionMode_t>(REDUCTION_TYPE)>(
        input,
        target,
        doutput,
        dinput,
        dtarget,
        alpha,
        gamma,
        size,
        input_tv,
        target_tv,
        doutput_tv,
        dinput_tv,
        dtarget_tv);
}
