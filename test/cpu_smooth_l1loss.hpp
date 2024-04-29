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
#ifndef GUARD_CPU_SMOOTH_L1LOSS_HPP
#define GUARD_CPU_SMOOTH_L1LOSS_HPP

#include "tensor_holder.hpp"
#include <miopen/tensor_view_5d.hpp>

template <class T>
void cpu_smooth_l1loss_unreduced_forward(tensor<T> input,
                                         tensor<T> target,
                                         tensor<T>& ref_output,
                                         float beta)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv(input.desc);
    auto T_tv = get_inner_expanded_tv(target.desc);
    auto O_tv = get_inner_expanded_tv(ref_output.desc);

    auto size = ref_output.desc.GetElementSize();
    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, O_tv);

        uint64_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        uint64_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);
        uint64_t Oidx = TV5D_IDX(O_tv, n[0], n[1], n[2], n[3], n[4]);

        auto diff        = abs(input[Iidx] - target[Tidx]);
        ref_output[Oidx] = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;
    });
}

template <class T>
void cpu_smooth_l1loss_reduced_forward(
    tensor<T> input, tensor<T> target, tensor<T>& ref_output, float beta, float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv(input.desc);
    auto T_tv = get_inner_expanded_tv(target.desc);

    auto size = input.desc.GetElementSize();
    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);

        uint64_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        uint64_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

        auto diff = abs(input[Iidx] - target[Tidx]);
        ref_output[0] += diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta / divisor;
    });
}

#endif // GUARD_CPU_SMOOTH_L1LOSS_HPP
