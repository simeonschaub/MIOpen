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
#ifndef GUARD_CPU_L1LOSS_HPP
#define GUARD_CPU_L1LOSS_HPP

#include "ford.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include "tensor_holder.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef LOCAL_SIZE_REDUCE
#define LOCAL_SIZE_REDUCE 1024
#endif

template <class T>
void cpu_l1loss_reduced_forward(tensor<T> input,
                                tensor<T> target,
                                tensor<T>& ref_output,
                                tensor<T>& ref_workspace,
                                miopenLossReductionMode_t reduction)
{
    auto inputSize = input.desc.GetElementSize();
    size_t divisor = (reduction == MIOPEN_LOSS_REDUCTION_SUM) ? 1 : inputSize;

    // Phase 1: Calc loss for each element (unreduced)
    par_ford(inputSize)([&](size_t i) { ref_workspace[i] = abs(input[i] - target[i]) / divisor; });

    /* Phase 2: Reduce */
    const int local_size = LOCAL_SIZE_REDUCE;
    int offset_a         = 0;
    int offset_b         = inputSize;
    size_t _size         = inputSize;
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

#endif // GUARD_CPU_L1LOSS_HPP
