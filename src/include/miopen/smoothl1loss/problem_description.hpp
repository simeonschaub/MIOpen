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
        if(iDesc.GetSize() != tDesc.GetSize())
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor sizes do not match.");
#else
            return false;
#endif
        for(int32_t i = 0; i < iDesc.GetSize(); ++i)
        {
            if(iDesc.GetLengths()[i] != tDesc.GetLengths()[i])
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor sizes do not match.");
#else
                return false;
#endif
            }
        }
        if(reduction == MIOPEN_LOSS_NO_REDUCTION)
        {
            if(iDesc.GetSize() != oDesc.GetSize())
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor sizes do not match.");
#else
                return false;
#endif
            }
            for(int32_t i = 0; i < iDesc.GetSize(); ++i)
            {
                if(iDesc.GetLengths()[i] != oDesc.GetLengths()[i])
                {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                    MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor sizes do not match.");
#else
                    return false;
#endif
                }
            }
        }
        return true;
    }

    bool IsRightStride() const
    {
        auto isRightStride = [](TensorDescriptor td) {
            auto lengths = td.GetLengths();
            auto strides = td.GetStrides();
            std::vector<std::pair<size_t, size_t>> p;
            p.reserve(td.GetSize());
            std::transform(lengths.begin(),
                           lengths.end(),
                           strides.begin(),
                           std::back_inserter(p),
                           [](size_t a, size_t b) { return std::make_pair(a, b); });
            std::sort(p.begin(), p.end(), [](auto a, auto b) { return a.second < b.second; });
            for(int i = 1; i < p.size(); ++i)
            {
                if(p[i].second != p[i - 1].first * p[i - 1].second)
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                    MIOPEN_THROW(miopenStatusBadParm,
                                 "Smooth L1Loss: Tensor strides do not match.");
#else
                    return false;
#endif
            }
            return true;
        };
        return isRightStride(iDesc) && isRightStride(tDesc) && isRightStride(oDesc);
    }

    bool IsSameStride() const
    {
        if(iDesc.GetSize() != tDesc.GetSize())
            return false;
        for(int32_t i = 0; i < iDesc.GetSize(); ++i)
            if(iDesc.GetStrides()[i] != tDesc.GetStrides()[i])
                return false;
        if(reduction == MIOPEN_LOSS_NO_REDUCTION)
        {
            if(iDesc.GetSize() != oDesc.GetSize())
                return false;
            for(int32_t i = 0; i < iDesc.GetSize(); ++i)
                if(iDesc.GetStrides()[i] != oDesc.GetStrides()[i])
                    return false;
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        auto isContiguous = [](TensorDescriptor td) {
            size_t s = 1;
            for(int i = td.GetSize() - 1; i >= 0; --i)
            {
                if(s != td.GetStrides()[i])
                    return false;
                s *= td.GetLengths()[i];
            }
            return true;
        };
        return isContiguous(iDesc) && isContiguous(tDesc) && isContiguous(oDesc);
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

struct UnreducedProblemDescription : ProblemDescriptionBase
{
    UnreducedProblemDescription(const TensorDescriptor& iDesc_,
                                const TensorDescriptor& tDesc_,
                                const TensorDescriptor& oDesc_,
                                float beta_)
        : iDesc(iDesc_), tDesc(tDesc_), oDesc(oDesc_), beta(beta_)
    {
    }

    const TensorDescriptor& GetIDesc() const { return iDesc; }
    const TensorDescriptor& GetTDesc() const { return tDesc; }
    const TensorDescriptor& GetODesc() const { return oDesc; }

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
        if(iDesc.GetSize() != tDesc.GetSize() || iDesc.GetSize() != oDesc.GetSize())
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor sizes do not match.");
#else
            return false;
#endif
        for(int32_t i = 0; i < iDesc.GetSize(); ++i)
        {
            if(iDesc.GetLengths()[i] != tDesc.GetLengths()[i] ||
               iDesc.GetLengths()[i] != oDesc.GetLengths()[i])
            {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor sizes do not match.");
#else
                return false;
#endif
            }
        }
        return true;
    }

    bool IsRightStride() const
    {
        auto isRightStride = [](TensorDescriptor td) {
            auto lengths = td.GetLengths();
            auto strides = td.GetStrides();
            std::vector<std::pair<size_t, size_t>> p;
            p.reserve(td.GetSize());
            std::transform(lengths.begin(),
                           lengths.end(),
                           strides.begin(),
                           std::back_inserter(p),
                           [](size_t a, size_t b) { return std::make_pair(a, b); });
            std::sort(p.begin(), p.end(), [](auto a, auto b) { return a.second < b.second; });
            for(int i = 1; i < p.size(); ++i)
            {
                if(p[i].second != p[i - 1].first * p[i - 1].second)
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                    MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Invalid tensor strides.");
#else
                    return false;
#endif
            }
            return true;
        };
        return isRightStride(iDesc) && isRightStride(tDesc) && isRightStride(oDesc);
    }

    bool IsSameStride() const
    {
        if(iDesc.GetSize() != tDesc.GetSize() || iDesc.GetSize() != oDesc.GetSize())
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
            MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor strides do not match.");
#else
            return false;
#endif
        for(int32_t i = 0; i < iDesc.GetSize(); ++i)
        {
            if(iDesc.GetStrides()[i] != tDesc.GetStrides()[i] ||
               iDesc.GetStrides()[i] != oDesc.GetStrides()[i])
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Tensor strides do not match.");
#else
                return false;
#endif
        }
        return true;
    }

    bool IsAllContiguous() const
    {
        auto isContiguous = [](TensorDescriptor td) {
            size_t s = 1;
            for(int i = td.GetSize() - 1; i >= 0; --i)
            {
                if(s != td.GetStrides()[i])
                {
#if MIOPEN_BUILD_DEV || !MIOPEN_NDEBUG
                    MIOPEN_THROW(miopenStatusBadParm, "Smooth L1Loss: Non-contiguous Tensor.");
#else
                    return false;
#endif
                }
                s *= td.GetLengths()[i];
            }
            return true;
        };
        return isContiguous(iDesc) && isContiguous(tDesc) && isContiguous(oDesc);
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    TensorDescriptor iDesc;
    TensorDescriptor tDesc;
    TensorDescriptor oDesc;
    float beta;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace smoothl1loss

} // namespace miopen
