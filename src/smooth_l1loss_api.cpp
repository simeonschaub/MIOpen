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

#include <miopen/smooth_l1loss.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

inline std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v)
{
    os << '{';
    for(int i = 0; i < v.size(); ++i)
    {
        if(i != 0)
            os << ',';
        os << v[i];
    }
    os << '}';
    return os;
}

static void LogCmdSmoothL1Loss(const miopenTensorDescriptor_t iDesc,
                               const miopenTensorDescriptor_t tDesc,
                               const float beta,
                               const float divisor,
                               bool is_fwd)
{
    if(miopen::IsLoggingCmd())
    {
        std::stringstream ss;
        auto dtype = miopen::deref(iDesc).GetType();
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

        MIOPEN_LOG_FUNCTION(iDesc, tDesc);
        ss << " -n " << miopen::deref(iDesc).GetLengths()[0];
        ss << " -T " << miopen::deref(iDesc).GetLengths();
        ss << " -Si " << miopen::deref(iDesc).GetStrides();
        ss << " -St " << miopen::deref(tDesc).GetStrides();
        ss << " -F " << ((is_fwd) ? "1" : "2") << " -b " << beta << " -d " << divisor;

        MIOPEN_LOG_DRIVER_CMD(ss.str());
    }
}

extern "C" miopenStatus_t
miopenGetSmoothL1LossReducedForwardWorkspaceSize(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t iDesc,
                                                 const miopenTensorDescriptor_t tDesc,
                                                 const miopenTensorDescriptor_t oDesc,
                                                 size_t* sizeInBytes)
{

    MIOPEN_LOG_FUNCTION(handle, iDesc, tDesc, oDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) =
            miopen::GetSmoothL1LossReducedForwardWorkspaceSize(miopen::deref(handle),
                                                               miopen::deref(iDesc),
                                                               miopen::deref(tDesc),
                                                               miopen::deref(oDesc));
    });
}

extern "C" miopenStatus_t miopenSmoothL1LossReducedForward(miopenHandle_t handle,
                                                           void* workspace,
                                                           size_t workspaceSizeInBytes,
                                                           const miopenTensorDescriptor_t iDesc,
                                                           const void* i,
                                                           const miopenTensorDescriptor_t tDesc,
                                                           const void* t,
                                                           const miopenTensorDescriptor_t oDesc,
                                                           void* o,
                                                           const float beta,
                                                           const float divisor)
{
    MIOPEN_LOG_FUNCTION(
        handle, workspace, workspaceSizeInBytes, iDesc, i, tDesc, t, oDesc, o, beta, divisor);

    LogCmdSmoothL1Loss(iDesc, tDesc, beta, divisor, true);
    return miopen::try_([&] {
        miopen::SmoothL1LossReducedForward(miopen::deref(handle),
                                           DataCast(workspace),
                                           workspaceSizeInBytes,
                                           miopen::deref(iDesc),
                                           DataCast(i),
                                           miopen::deref(tDesc),
                                           DataCast(t),
                                           miopen::deref(oDesc),
                                           DataCast(o),
                                           beta,
                                           divisor);
    });
}

extern "C" miopenStatus_t miopenSmoothL1LossReducedBackward(miopenHandle_t handle,
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
                                                            const float beta,
                                                            const float divisor)
{
    MIOPEN_LOG_FUNCTION(
        handle, iDesc, i, tDesc, t, doDesc, dO, diDesc, dI, dtDesc, dT, beta, divisor);

    LogCmdSmoothL1Loss(iDesc, tDesc, beta, divisor, false);
    return miopen::try_([&] {
        miopen::SmoothL1LossReducedBackward(miopen::deref(handle),
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
                                            beta,
                                            divisor);
    });
}
