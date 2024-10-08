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
void cpu_smoothl1loss_forward(tensor<T> input,
                              tensor<T> target,
                              tensor<T>& ref_output,
                              tensor<T>& ref_workspace,
                              float beta,
                              miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv<5>(input.desc);
    auto T_tv = get_inner_expanded_tv<5>(target.desc);
    auto O_tv = get_inner_expanded_tv<5>(ref_output.desc);

    auto size = input.desc.GetElementSize();

    /* Phase 1: Calc loss for each element. */
    par_ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);
        auto diff                = abs(input[Iidx] - target[Tidx]);
        auto loss                = (diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta);
        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
            loss /= size;
        ref_workspace[Iidx] = loss;
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
            ref_output[O_tv.get_tensor_view_idx(tensor_layout)] = loss;
    });

    /* Phase 2: Reduce */
    const int local_size = 256;
    int offset_a         = 0;
    int offset_b         = size;
    size_t _size         = size;
    do
    {
        for(int i = 0; i < _size; i += local_size)
        {
            T shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? ref_workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < offset; ++j)
                    shared[j] += shared[j + offset];
            if(_size <= local_size)
                ref_output[0] = shared[0];
            else
                ref_workspace[offset_b + i / local_size] = shared[0];
        }
        std::swap(offset_a, offset_b);
        _size = (_size + local_size - 1) / local_size;
    } while(_size > 1);
}

template <class T>
void cpu_smoothl1loss_backward(tensor<T> input,
                               tensor<T> target,
                               tensor<T> dO,
                               tensor<T>& ref_dI,
                               tensor<T>& ref_dT,
                               float beta,
                               miopenLossReductionMode_t reduction)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv  = get_inner_expanded_tv<5>(input.desc);
    auto T_tv  = get_inner_expanded_tv<5>(target.desc);
    auto dO_tv = get_inner_expanded_tv<5>(dO.desc);
    auto dI_tv = get_inner_expanded_tv<5>(ref_dI.desc);
    auto dT_tv = get_inner_expanded_tv<5>(ref_dT.desc);

    auto size = input.desc.GetElementSize();

    par_ford(size)([&](size_t i) {
        const auto tensor_layout = tensor_layout_t<5>(I_tv, i);
        const uint64_t Iidx      = I_tv.get_tensor_view_idx(tensor_layout);
        const uint64_t Tidx      = T_tv.get_tensor_view_idx(tensor_layout);

        T sub  = input[Iidx] - target[Tidx];
        T grad = static_cast<T>(0.0f);

        if(fabs(sub) < beta)
            grad = sub / beta *
                   dO[reduction == MIOPEN_LOSS_REDUCTION_NONE
                          ? dO_tv.get_tensor_view_idx(tensor_layout)
                          : 0];
        else
            grad = (sub >= 0 ? 1.0f : -1.0f) * dO[reduction == MIOPEN_LOSS_REDUCTION_NONE
                                                      ? dO_tv.get_tensor_view_idx(tensor_layout)
                                                      : 0];

        if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
            grad = grad / size;

        ref_dI[dI_tv.get_tensor_view_idx(tensor_layout)] = grad;
        ref_dT[dT_tv.get_tensor_view_idx(tensor_layout)] = -grad;
    });
}
