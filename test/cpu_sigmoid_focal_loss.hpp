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
#pragma once

#include "tensor_holder.hpp"
#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

template <class T>
void cpu_sigmoid_focal_loss_forward(tensor<T> input,
                                    tensor<T> target,
                                    tensor<T>& ref_output,
                                    float alpha,
                                    float gamma,
                                    miopenLossReductionMode_t reduction)
{
    auto input_tv  = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv = miopen::get_inner_expanded_tv<5>(target.desc);
    auto output_tv = miopen::get_inner_expanded_tv<5>(ref_output.desc);
    size_t size    = input.desc.GetElementSize();

    std::vector<double> buffer;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        buffer.assign(size, 0);

    par_ford(size)([&](size_t idx) {
        tensor_layout_t<5> tensor_layout(input_tv, idx);

        float i = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);
        float t = static_cast<float>(target[target_tv.get_tensor_view_idx(tensor_layout)]);

        float sig    = 1 / (1 + std::exp(-i));
        float ceLoss = -(t * std::log(sig) + (1 - t) * std::log(1 - sig));
        float sigT   = sig * t + (1 - sig) * (1 - t);
        float loss   = ceLoss * std::pow(1 - sigT, gamma);

        if(alpha >= 0)
        {
            float alphaT = alpha * t + (1 - alpha) * (1 - t);
            loss         = alphaT * loss;
        }

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
            ref_output[output_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(loss);
        else
            buffer[idx] = loss;
    });

    auto loss_sum = std::accumulate(buffer.begin(), buffer.end(), 0.0);

    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        loss_sum /= size;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        ref_output[0] = static_cast<T>(loss_sum);
}

template <class T>
void cpu_sigmoid_focal_loss_backward(tensor<T> input,
                                     tensor<T> target,
                                     tensor<T> doutput,
                                     tensor<T>& ref_dinput,
                                     tensor<T>& ref_dtarget,
                                     float alpha,
                                     float gamma,
                                     miopenLossReductionMode_t reduction)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(input.desc);
    auto target_tv  = miopen::get_inner_expanded_tv<5>(target.desc);
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(doutput.desc);
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(ref_dinput.desc);
    auto dtarget_tv = miopen::get_inner_expanded_tv<5>(ref_dtarget.desc);

    size_t size = input.desc.GetElementSize();

    par_ford(size)([&](size_t idx) {
        tensor_layout_t<5> tensor_layout(input_tv, idx);

        float i  = static_cast<float>(input[input_tv.get_tensor_view_idx(tensor_layout)]);
        float t  = static_cast<float>(target[target_tv.get_tensor_view_idx(tensor_layout)]);
        float dO = static_cast<float>(doutput[0]);
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
            dO = static_cast<float>(doutput[doutput_tv.get_tensor_view_idx(tensor_layout)]);
        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
            dO /= size;

        float p       = 1 / (1 + std::exp(-i));
        float ceLoss  = -(t * std::log(p) + (1 - t) * std::log(1 - p));
        float pT      = p * t + (1 - p) * (1 - t);
        float powPt   = std::pow(1 - pT, gamma);
        float alpha_t = alpha * t + (1 - alpha) * (1 - t);

        if(ref_dinput.data.size() > 0)
        {
            float dpdi      = std::exp(-i) / std::pow(1 + std::exp(-i), 2);
            float dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
            float dpowptdi  = gamma * std::pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

            // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
            float dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
            float grad = dO * dLdi;

            if(alpha >= 0)
                grad *= alpha_t;

            ref_dinput[dinput_tv.get_tensor_view_idx(tensor_layout)] =
                static_cast<T>(static_cast<float>(grad));
        }

        if(ref_dtarget.data.size() > 0)
        {
            float dcelossdt = -std::log(p) + std::log(1 - p);
            float dpowptdt  = gamma * std::pow(1 - pT, gamma - 1) * (1 - 2 * p);
            // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
            float dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
            float gradTarget = dO * dLdt;

            if(alpha >= 0)
            {
                // alpha_t * dL/dt + dalpha_t/dt * L
                gradTarget = dO * (alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt);
            }
            ref_dtarget[dtarget_tv.get_tensor_view_idx(tensor_layout)] = static_cast<T>(gradTarget);
        }
    });
}

template <typename TIO>
float get_tolerance()
{
    float tolerance = std::is_same<TIO, float>::value ? 1.5e-6 : 8.2e-3;
    // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
    if(std::is_same<TIO, bfloat16>::value)
        tolerance *= 8.0;
    return tolerance;
}
