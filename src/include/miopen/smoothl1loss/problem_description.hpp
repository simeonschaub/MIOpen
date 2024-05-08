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

#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace smoothl1loss {

bool checkSameType(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameStride(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkRightStride(const TensorDescriptor& x);
bool checkContiguous(const TensorDescriptor& x);

struct SmoothL1LossFwdProblemDescription : ProblemDescriptionBase
{
    SmoothL1LossFwdProblemDescription(const TensorDescriptor& iDesc_,
                                      const TensorDescriptor& tDesc_,
                                      const TensorDescriptor& oDesc_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }

    bool IsSameType() const { return checkSameType(iDesc, tDesc); }

    bool IsRightLength() const { return checkSameLength(iDesc, tDesc); }

    bool IsRightStride() const
    {
        return checkRightStride(iDesc) && checkRightStride(tDesc) && checkRightStride(oDesc);
    }

    bool IsSameStride() const { return checkSameStride(iDesc, tDesc); }

    bool IsAllContiguous() const
    {
        return checkContiguous(iDesc) && checkContiguous(tDesc) && checkContiguous(oDesc);
    }

protected:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor oDesc;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct ReducedForwardProblemDescription : SmoothL1LossFwdProblemDescription
{
    ReducedForwardProblemDescription(const TensorDescriptor& iDesc_,
                                     const TensorDescriptor& tDesc_,
                                     const TensorDescriptor& oDesc_)
        : SmoothL1LossFwdProblemDescription(iDesc_, tDesc_, oDesc_)
    {
    }

    bool IsRightLength() const
    {
        if(!SmoothL1LossFwdProblemDescription::IsRightLength())
            return false;
        if(oDesc.GetSize() != 1 || oDesc.GetLengths()[0] != 1)
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Output Tensor size must be (1).");
#else
            return false;
#endif
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;
};

struct UnreducedForwardProblemDescription : SmoothL1LossFwdProblemDescription
{
    UnreducedForwardProblemDescription(const TensorDescriptor& iDesc_,
                                       const TensorDescriptor& tDesc_,
                                       const TensorDescriptor& oDesc_)
        : SmoothL1LossFwdProblemDescription(iDesc_, tDesc_, oDesc_)
    {
    }

    bool IsRightLength() const
    {
        if(!SmoothL1LossFwdProblemDescription::IsRightLength())
            return false;
        return checkSameLength(iDesc, oDesc);
    }

    bool IsSameStride() const
    {
        if(!SmoothL1LossFwdProblemDescription::IsSameStride())
            return false;
        return checkSameStride(iDesc, oDesc);
    }

    NetworkConfig MakeNetworkConfig() const override;
};

struct SmoothL1LossBwdProblemDescription : ProblemDescriptionBase
{
    SmoothL1LossBwdProblemDescription(const TensorDescriptor& iDesc_,
                                      const TensorDescriptor& tDesc_,
                                      const TensorDescriptor& doDesc_,
                                      const TensorDescriptor& diDesc_,
                                      const TensorDescriptor& dtDesc_)
        : iDesc(iDesc_), tDesc(tDesc_), doDesc(doDesc_), diDesc(diDesc_), dtDesc(dtDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetDODesc() const { return doDesc; }
    const TensorDescriptor& GetDIDesc() const { return diDesc; }
    const TensorDescriptor& GetDTDesc() const { return dtDesc; }

    bool IsSameType() const
    {
        return checkSameType(iDesc, tDesc) && checkSameType(iDesc, diDesc) &&
               checkSameType(tDesc, dtDesc);
    }

    bool IsRightLength() const
    {
        return checkSameLength(iDesc, tDesc) && checkSameLength(iDesc, diDesc) &&
               checkSameLength(tDesc, dtDesc);
    }

    bool IsRightStride() const
    {
        return checkRightStride(iDesc) && checkRightStride(tDesc) && checkRightStride(doDesc) &&
               checkRightStride(diDesc) && checkRightStride(dtDesc);
    }

    bool IsAllContiguous() const
    {
        return checkContiguous(iDesc) && checkContiguous(tDesc) && checkContiguous(doDesc) &&
               checkContiguous(diDesc) && checkContiguous(dtDesc);
    }

protected:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor doDesc;
    TensorDescriptor diDesc;
    TensorDescriptor dtDesc;

    NetworkConfig MakeBackwardNetworkConfig() const;
};

struct ReducedBackwardProblemDescription : SmoothL1LossBwdProblemDescription
{
    ReducedBackwardProblemDescription(const TensorDescriptor& iDesc_,
                                      const TensorDescriptor& tDesc_,
                                      const TensorDescriptor& doDesc_,
                                      const TensorDescriptor& diDesc_,
                                      const TensorDescriptor& dtDesc_)
        : SmoothL1LossBwdProblemDescription(iDesc_, tDesc_, doDesc_, diDesc_, dtDesc_)
    {
    }

    bool IsRightLength() const
    {
        if(!SmoothL1LossBwdProblemDescription::IsRightLength())
            return false;
        if(doDesc.GetSize() != 1 || doDesc.GetLengths()[0] != 1)
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm,
                         "Smooth L1Loss: Output Gradient Tensor size must be (1).");
#else
            return false;
#endif
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;
};

} // namespace smoothl1loss

} // namespace miopen
