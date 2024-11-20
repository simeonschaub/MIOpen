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

#include <miopen/miopen.h>
#include <miopen/l1loss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenGetL1LossForwardWorkspaceSize(miopenHandle_t handle,
                                                              miopenLossReductionMode_t reduction,
                                                              const miopenTensorDescriptor_t iDesc,
                                                              const miopenTensorDescriptor_t tDesc,
                                                              const miopenTensorDescriptor_t oDesc,
                                                              size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, reduction, iDesc, tDesc, oDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetL1LossForwardWorkspaceSize(miopen::deref(handle),
                                                                           reduction,
                                                                           miopen::deref(iDesc),
                                                                           miopen::deref(tDesc),
                                                                           miopen::deref(oDesc));
    });
}

extern "C" miopenStatus_t miopenL1LossForward(miopenHandle_t handle,
                                              miopenLossReductionMode_t reduction,
                                              void* workspace,
                                              size_t workspaceSizeInBytes,
                                              const miopenTensorDescriptor_t iDesc,
                                              const void* i,
                                              const miopenTensorDescriptor_t tDesc,
                                              const void* t,
                                              const miopenTensorDescriptor_t oDesc,
                                              void* o)
{
    MIOPEN_LOG_FUNCTION(
        handle, reduction, workspace, workspaceSizeInBytes, iDesc, i, tDesc, t, oDesc, o);

    return miopen::try_([&] {
        miopen::L1LossForward(miopen::deref(handle),
                              reduction,
                              DataCast(workspace),
                              workspaceSizeInBytes,
                              miopen::deref(iDesc),
                              DataCast(i),
                              miopen::deref(tDesc),
                              DataCast(t),
                              miopen::deref(oDesc),
                              DataCast(o));
    });
}
