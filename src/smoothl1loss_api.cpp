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
#include <miopen/smoothl1loss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int input = 0; input < v.size(); ++input)
    {
        if(input != 0)
            os << ',';
        os << v[input];
    }
    os << '}';
    return os;
}

static void LogCmdSmoothL1Loss(const miopenTensorDescriptor_t inputDesc,
                               const miopenTensorDescriptor_t targetDesc,
                               const float beta,
                               const miopenLossReductionMode_t reduction,
                               bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(inputDesc).GetType();
        if(dtype == miopenHalf)
        {
            ss << "smoothl1lossfp16";
        }
        else if(dtype == miopenFloat)
        {
            ss << "smoothl1lossfp32";
        }
        else if(dtype == miopenBFloat16)
        {
            ss << "smoothl1lossbfp16";
        }

        MIOPEN_LOG_FUNCTION(inputDesc, targetDesc);
        ss << " -n " << miopen::deref(inputDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(inputDesc).GetLengths();
        ss << " -Si " << miopen::deref(inputDesc).GetStrides();
        ss << " -St " << miopen::deref(targetDesc).GetStrides();
        ss << " -F " << ((is_fwd) ? "1" : "2") << " -b " << beta << " -r " << reduction;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetSmoothL1LossForwardWorkspaceSize(miopenHandle_t handle,
                                          const miopenTensorDescriptor_t inputDesc,
                                          const miopenTensorDescriptor_t outputDesc,
                                          const miopenLossReductionMode_t reduction,
                                          size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, inputDesc, outputDesc, reduction, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::GetSmoothL1LossForwardWorkspaceSize(
            miopen::deref(handle), miopen::deref(inputDesc), miopen::deref(outputDesc), reduction);
    });
}

extern "C" miopenStatus_t miopenSmoothL1LossForward(miopenHandle_t handle,
                                                    void* workspace,
                                                    size_t workspaceSizeInBytes,
                                                    const miopenTensorDescriptor_t inputDesc,
                                                    const void* input,
                                                    const miopenTensorDescriptor_t targetDesc,
                                                    const void* target,
                                                    const miopenTensorDescriptor_t outputDesc,
                                                    void* output,
                                                    const float beta,
                                                    const miopenLossReductionMode_t reduction)
{
    MIOPEN_LOG_FUNCTION(handle,
                        workspace,
                        workspaceSizeInBytes,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        outputDesc,
                        output,
                        beta,
                        reduction);

    LogCmdSmoothL1Loss(inputDesc, targetDesc, beta, reduction, true);
    return miopen::try_([&] {
        miopen::SmoothL1LossForward(miopen::deref(handle),
                                    DataCast(workspace),
                                    workspaceSizeInBytes,
                                    miopen::deref(inputDesc),
                                    DataCast(input),
                                    miopen::deref(targetDesc),
                                    DataCast(target),
                                    miopen::deref(outputDesc),
                                    DataCast(output),
                                    beta,
                                    reduction);
    });
}

extern "C" miopenStatus_t miopenSmoothL1LossBackward(miopenHandle_t handle,
                                                     const miopenTensorDescriptor_t inputDesc,
                                                     const void* input,
                                                     const miopenTensorDescriptor_t targetDesc,
                                                     const void* target,
                                                     const miopenTensorDescriptor_t doutputDesc,
                                                     const void* doutput,
                                                     const miopenTensorDescriptor_t dinputDesc,
                                                     void* dinput,
                                                     const miopenTensorDescriptor_t dtargetDesc,
                                                     void* dtarget,
                                                     const float beta,
                                                     const miopenLossReductionMode_t reduction)
{
    MIOPEN_LOG_FUNCTION(handle,
                        inputDesc,
                        input,
                        targetDesc,
                        target,
                        doutputDesc,
                        doutput,
                        dinputDesc,
                        dinput,
                        dtargetDesc,
                        dtarget,
                        beta,
                        reduction);

    LogCmdSmoothL1Loss(inputDesc, targetDesc, beta, reduction, false);
    return miopen::try_([&] {
        miopen::SmoothL1LossBackward(miopen::deref(handle),
                                     miopen::deref(inputDesc),
                                     DataCast(input),
                                     miopen::deref(targetDesc),
                                     DataCast(target),
                                     miopen::deref(doutputDesc),
                                     DataCast(doutput),
                                     miopen::deref(dinputDesc),
                                     DataCast(dinput),
                                     miopen::deref(dtargetDesc),
                                     DataCast(dtarget),
                                     beta,
                                     reduction);
    });
}
