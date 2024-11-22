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

#include <miopen/miopen.h>

#include "tensor_holder.hpp"
#include <cstddef>

template <class T>
void cpu_l1loss_reduced_forward(tensor<T> input,
                                tensor<T> target,
                                tensor<T>& ref_output,
                                miopenLossReductionMode_t reduction)
{
    auto inputSize = input.desc.GetElementSize();
    size_t divisor = (reduction == MIOPEN_LOSS_REDUCTION_SUM) ? 1 : inputSize;

    double output = 0.0;
    for(size_t i = 0; i < inputSize; i++)
    {
        float diff = abs(static_cast<float>(input[i]) - static_cast<float>(target[i]));
        output += diff;
    }
    ref_output[0] = output / divisor;
}
