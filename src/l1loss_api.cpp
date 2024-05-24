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

#include "miopen/miopen.h"
#include <miopen/l1loss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

static void LogCmdL1Loss(const miopenTensorDescriptor_t iDesc,
                         const miopenL1LossReduction_t reduction,
                         bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(iDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "l1lossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "l1lossfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "l1lossbfp16";
        }

        int32_t size = {0};
        miopenGetTensorDescriptorSize(iDesc, &size);
        ss << " -n " << miopen::deref(iDesc).GetLengths()[0];
        if(size == 5)
        {
            ss << " -c " << miopen::deref(iDesc).GetLengths()[1] << " -D "
               << miopen::deref(iDesc).GetLengths()[2] << " -H "
               << miopen::deref(iDesc).GetLengths()[3] << " -W "
               << miopen::deref(iDesc).GetLengths()[4];
        }
        else if(size == 4)
        {
            ss << " -c " << miopen::deref(iDesc).GetLengths()[1] << " -H "
               << miopen::deref(iDesc).GetLengths()[2] << " -W "
               << miopen::deref(iDesc).GetLengths()[3];
        }
        else if(size == 3)
        {
            ss << " -c " << miopen::deref(iDesc).GetLengths()[1] << " -W "
               << miopen::deref(iDesc).GetLengths()[2];
        }
        else if(size == 2)
        {
            ss << " -c " << miopen::deref(iDesc).GetLengths()[1];
        }

        ss << " -F " << ((is_fwd) ? "1" : "2") << " -r " << reduction;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t miopenGetL1LossForwardWorkspaceSize(miopenHandle_t handle,
                                                              miopenL1LossReduction_t reduction,
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
                                              miopenL1LossReduction_t reduction,
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

    LogCmdL1Loss(iDesc, reduction, true);
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

extern "C" miopenStatus_t miopenL1LossBackward(miopenHandle_t handle,
                                                            const miopenTensorDescriptor_t iDesc,
                                                            const void* i,
                                                            const miopenTensorDescriptor_t tDesc,
                                                            const void* t,
                                                            const miopenTensorDescriptor_t doDesc,
                                                            const void* dO,
                                                            const miopenTensorDescriptor_t diDesc,
                                                            void* dI,
                                                            const miopenTensorDescriptor_t dtDesc,
                                                            void* dT,
                                                            miopenL1LossReduction_t reduction)
{
    MIOPEN_LOG_FUNCTION(
        handle, iDesc, i, tDesc, t, doDesc, dO, diDesc, dI, dtDesc, dT, reduction);

    LogCmdL1Loss(iDesc, reduction, false);
    return miopen::try_([&] {
        miopen::L1LossBackward(miopen::deref(handle),
                                            miopen::deref(iDesc),
                                            DataCast(i),
                                            miopen::deref(tDesc),
                                            DataCast(t),
                                            miopen::deref(doDesc),
                                            DataCast(dO),
                                            miopen::deref(diDesc),
                                            DataCast(dI),
                                            miopen::deref(dtDesc),
                                            DataCast(dT),
                                            reduction);
    });
}
