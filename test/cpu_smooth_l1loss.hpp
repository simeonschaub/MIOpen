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
template <class T>
void cpu_smooth_l1loss_forward(tensor<T> input,
                               tensor<T> target,
                               tensor<T>& ref_output,
                               miopenLossReduction_t reduction,
                               float beta)
{
    auto dims   = input.desc.GetLengths();
    size_t size = std::accumulate(dims.begin(), dims.end(), 1L, std::multiplies<int64_t>());

    auto loss_no_reduce = [&]() {
        par_ford(size)([&](size_t i) {
            auto diff     = abs(input[i] - target[i]);
            ref_output[i] = diff < beta ? 0.5f * diff * diff / beta : diff - 0.5f * beta;
        });
    };

    auto loss_mean_reduce = [&]() { std::cout << "Unsupported Mean Reduction"; };

    auto loss_sum_reduce = [&]() { std::cout << "Unsupported Sum Reduction"; };

    switch(reduction)
    {
    case MIOPEN_LOSS_MEAN_REDUCTION: loss_mean_reduce(); break;
    case MIOPEN_LOSS_SUM_REDUCTION: loss_sum_reduce(); break;
    default: loss_no_reduce(); break;
    }
}

#endif // GUARD_CPU_SMOOTH_L1LOSS_HPP
