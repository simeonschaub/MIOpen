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

struct ProblemDescription : ProblemDescriptionBase
{
    ProblemDescription(miopenLossReduction_t reduction_,
                       const TensorDescriptor& iDesc_,
                       const TensorDescriptor& tDesc_,
                       const TensorDescriptor& oDesc_,
                       float beta_)
        : reduction(reduction_), iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_), beta(beta_)
    {
    }

    ProblemDescription(miopenLossReduction_t reduction_,
                       const TensorDescriptor& iDesc_,
                       const TensorDescriptor& tDesc_,
                       const TensorDescriptor& oDesc_)
        : reduction(reduction_), iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }
    miopenLossReduction_t GetReduction() const { return reduction; }

    bool IsSameType() const
    {
        if(iDesc.GetType() != tDesc.GetType())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Reduce: Tensor types do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    bool IsRightLength() const
    {
        if(iDesc.GetElementSize() != tDesc.GetElementSize() ||
           iDesc.GetElementSize() != oDesc.GetElementSize())
        {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor sizes do not match.");
#else
            return false;
#endif
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    miopenLossReduction_t reduction;
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor oDesc;
    float beta;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace smoothl1loss

} // namespace miopen
