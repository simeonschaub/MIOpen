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

template <class T>
void cpu_l1loss_reduced_forward(tensor<T> input,
                                tensor<T> target,
                                tensor<T>& ref_output,
                                tensor<T>& ref_workspace,
                                miopenL1LossReduction_t reduction)
{
    auto inputSize = input.desc.GetElementSize();
    size_t divisor = (reduction == MIOPEN_L1LOSS_SUM_REDUCTION) ? 1 : inputSize;

    // Phase 1: Calc loss for each element (unreduced)
    par_ford(inputSize)([&](size_t i) {
        ref_workspace[i] = abs(input[i] - target[i]) / divisor; 
    });

    /* Phase 2: Reduce */
    const int local_size = 256;
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

    std::cout << "find finite " << std::endl;
    par_ford(inputSize)([&](size_t i) {
        if (!std::isfinite(ref_workspace[i])) {
            std::cout << "index = " << i << std::endl;
        }
    });


    //ref_output[0] = static_cast<T>(res);
    std::cout << ref_workspace[0] << std::endl;
    std::cout << ref_workspace[inputSize / 2] << std::endl;
    std::cout << "divisor = " << divisor << std::endl;
    std::cout << "input size = " << inputSize << std::endl;
    std::cout << "res = " << ref_output[0] << std::endl;
}

/*
template <class T>
void cpu_l1loss_reduced_backward(tensor<T> input,
                                 tensor<T> target,
                                 tensor<T> dO,
                                 tensor<T>& ref_dI,
                                 tensor<T>& ref_dT,
                                 float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv  = get_inner_expanded_tv(input.desc);
    auto T_tv  = get_inner_expanded_tv(target.desc);
    auto dI_tv = get_inner_expanded_tv(ref_dI.desc);
    auto dT_tv = get_inner_expanded_tv(ref_dT.desc);

    auto size = input.desc.GetElementSize();

    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);

        size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

        T sub  = input[Iidx] - target[Tidx];
        T grad = static_cast<T>(0.0f);

        if(fabs(sub) < beta)
            grad = sub / beta * dO[0] / divisor;
        else
            grad = (sub >= 0 ? 1.0f : -1.0f) * dO[0] / divisor;

        ref_dI[TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4])] = grad;
        ref_dT[TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4])] = -grad;
    });
}
*/

#endif // GUARD_CPU_L1LOSS_HPP
