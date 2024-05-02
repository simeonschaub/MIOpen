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
void cpu_smooth_l1loss_reduced_forward(tensor<T> input,
                                       tensor<T> target,
                                       tensor<T>& ref_output,
                                       tensor<T>& ref_workspace,
                                       float beta,
                                       float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv = get_inner_expanded_tv(input.desc);
    auto T_tv = get_inner_expanded_tv(target.desc);

    auto size = input.desc.GetElementSize();

    /* Phase 1: Calc loss for each element. */
    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);

        uint64_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        uint64_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

        auto diff = abs(input[Iidx] - target[Tidx]);
        ref_workspace[Iidx] =
            (diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta) / divisor;
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
            float shared[local_size];
            for(int j = 0; j < local_size; ++j)
                shared[j] = i + j < _size ? ref_workspace[offset_a + i + j] : 0.0f;
            for(int offset = local_size / 2; offset > 0; offset >>= 1)
                for(int j = 0; j < local_size; ++j)
                    if(j < offset)
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

#endif // GUARD_CPU_SMOOTH_L1LOSS_HPP
