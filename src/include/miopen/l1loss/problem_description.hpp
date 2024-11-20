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
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace l1loss {

struct FwdProblemDescription : ProblemDescriptionBase
{
    FwdProblemDescription(const TensorDescriptor& iDesc_,
                          const TensorDescriptor& tDesc_,
                          const TensorDescriptor& oDesc_,
                          miopenLossReductionMode_t reduction_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_), reduction(reduction_)
    {
        if(iDesc.GetNumDims() != tDesc.GetNumDims())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "L1Loss::ProblemDescription: Number of dimensions between input tensor "
                         "and target tensor do not match.");
        }

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            if(iDesc.GetNumDims() != oDesc.GetNumDims())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "L1Loss::ProblemDescription: Number of dimensions between input "
                             "tensor and output tensor do not match.");
            }
        }
        else
        {
            if(oDesc.GetNumDims() != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "L1Loss::ProblemDescription: Number of output tensor's dimension do "
                             "not equal 1 in case of reduction.");
            }
        }

        if(!IsSameType())
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "L1Loss::ProblemDescription: Input, target and output tensor have different "
                "data type.");
        }
    }

    miopenLossReductionMode_t GetReduction() const { return reduction; }
    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }

    bool IsSameType() const
    {
        if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != oDesc.GetType())
        {
            return false;
        }
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor oDesc;
    miopenLossReductionMode_t reduction;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace l1loss

} // namespace miopen
