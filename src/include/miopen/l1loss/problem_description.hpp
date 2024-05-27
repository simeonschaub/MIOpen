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

#include "miopen/miopen.h"
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <string>

namespace miopen {

struct NetworkConfig;

namespace l1loss {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkSameStride(const TensorDescriptor& x, const TensorDescriptor& y);
bool checkRightStride(const TensorDescriptor& x);
bool checkContiguous(const TensorDescriptor& x);

struct L1LossFwdProblemDescription : ProblemDescriptionBase
{
    L1LossFwdProblemDescription(const TensorDescriptor& iDesc_,
                                const TensorDescriptor& tDesc_,
                                const TensorDescriptor& oDesc_,
                                miopenL1LossReduction_t reduction_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_), reduction(reduction_)
    {
        if(iDesc.GetLengths().size() != tDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "L1Loss::ProblemDescription: Number of dimensions between input tensor "
                         "and target tensor do not match.");
        }

        if(reduction == MIOPEN_L1LOSS_NONE_REDUCTION)
        {
            if(iDesc.GetLengths().size() != oDesc.GetLengths().size())
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "L1Loss::ProblemDescription: Number of dimensions between input "
                             "tensor and output tensor do not match.");
            }
        }
        else
        {
            if(oDesc.GetLengths().size() != 1)
            {
                MIOPEN_THROW(miopenStatusBadParm,
                             "L1Loss::ProblemDescription: Number of output tensor's dimension do "
                             "not equal 1 in case of reduction.");
            }
        }
    }

    L1LossFwdProblemDescription(const TensorDescriptor& iDesc_,
                                const TensorDescriptor& tDesc_,
                                const TensorDescriptor& oDesc_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_)
    {
        if(iDesc.GetLengths().size() != tDesc.GetLengths().size())
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "L1Loss::ProblemDescription: Number of dimensions between input tensor "
                         "and target tensor do not match.");
        }

        if(oDesc.GetLengths().size() != 1)
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "L1Loss::ProblemDescription: Number of output tensor's dimension do not "
                         "equal 1 in case of reduction.");
        }
    }

    miopenL1LossReduction_t GetReduction() const { return reduction; }
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

    bool IsRightLength() const
    {
        if(!checkSameLength(iDesc, tDesc))
        {
            return false;
        }

        if(reduction == MIOPEN_L1LOSS_NONE_REDUCTION && !checkSameLength(iDesc, oDesc))
        {
            return false;
        }

        return true;
    }

    bool IsRightStride() const
    {
        if(!checkRightStride(iDesc) || !checkRightStride(tDesc) || !checkRightStride(oDesc))
        {
            return false;
        }
        return true;
    }

    bool IsSameStride() const
    {
        if(!checkSameStride(iDesc, tDesc))
        {
            return false;
        }

        if(reduction == MIOPEN_L1LOSS_NONE_REDUCTION && !checkSameStride(iDesc, oDesc))
        {
            return false;
        }

        return true;
    }

    bool IsAllPacked() const
    {
        if(!(iDesc.IsPacked() && tDesc.IsPacked() && oDesc.IsPacked()))
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
    miopenL1LossReduction_t reduction;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct L1LossBwdProblemDescription : ProblemDescriptionBase
{
    L1LossBwdProblemDescription(const TensorDescriptor& iDesc_,
                                const TensorDescriptor& tDesc_,
                                const TensorDescriptor& doDesc_,
                                const TensorDescriptor& diDesc_,
                                const TensorDescriptor& dtDesc_,
                                const miopenL1LossReduction_t reduction_)
        : iDesc(iDesc_),
          tDesc(tDesc_),
          doDesc(doDesc_),
          diDesc(diDesc_),
          dtDesc(dtDesc_),
          reduction(reduction_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetDODesc() const { return doDesc; }
    const TensorDescriptor& GetDIDesc() const { return diDesc; }
    const TensorDescriptor& GetDTDesc() const { return dtDesc; }
    const miopenL1LossReduction_t& GetReduction() const { return reduction; }

    bool IsSameType() const
    {
        if(iDesc.GetType() != tDesc.GetType() || iDesc.GetType() != diDesc.GetType() ||
           tDesc.GetType() != dtDesc.GetType())
        {
            return false;
        }
        return true;
    }

    bool IsRightLength() const
    {
        if(!checkSameLength(iDesc, tDesc) || !checkSameLength(iDesc, diDesc) ||
           !checkSameLength(tDesc, dtDesc))
        {
            return false;
        }
        return true;
    }

    bool IsRightStride() const
    {
        if(!checkRightStride(iDesc) || !checkRightStride(tDesc) || !checkRightStride(doDesc) ||
           !checkRightStride(diDesc) || !checkRightStride(dtDesc))
        {
            return false;
        }
        return true;
    }

    bool IsSameStride() const
    {
        if(!checkSameStride(iDesc, tDesc) || !checkSameStride(iDesc, diDesc) ||
           !checkSameStride(tDesc, dtDesc))
        {
            return false;
        }
        return true;
    }

    bool IsAllPacked() const
    {
        if(!(iDesc.IsPacked() && tDesc.IsPacked() && doDesc.IsPacked() && dtDesc.IsPacked() &&
             diDesc.IsPacked()))
        {
            return false;
        }

        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor doDesc;
    TensorDescriptor diDesc;
    TensorDescriptor dtDesc;
    miopenL1LossReduction_t reduction;

    NetworkConfig MakeBackwardNetworkConfig() const;
};

} // namespace l1loss

} // namespace miopen
