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

#include <miopen/l1loss/problem_description.hpp>
#include <miopen/names.hpp>

#include <sstream>

namespace miopen {

namespace l1loss {

bool checkSameLength(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetNumDims() != y.GetNumDims())
        return false;
    for(int32_t i = 0; i < x.GetNumDims(); ++i)
    {
        if(x.GetLengths()[i] != y.GetLengths()[i])
            return false;
    }
    return true;
}

bool checkSameStride(const TensorDescriptor& x, const TensorDescriptor& y)
{
    if(x.GetNumDims() != y.GetNumDims())
        return false;
    for(int32_t i = 0; i < x.GetNumDims(); ++i)
    {
        if(x.GetStrides()[i] != y.GetStrides()[i])
            return false;
    }
    return true;
}

bool checkRightStride(const TensorDescriptor& x)
{
    auto strides = x.GetStrides();
    auto lengths = x.GetLengths();
    std::vector<std::pair<size_t, size_t>> p;
    p.reserve(x.GetNumDims());
    std::transform(strides.begin(),
                   strides.end(),
                   lengths.begin(),
                   std::back_inserter(p),
                   [](size_t a, size_t b) { return std::make_pair(a, b); });
    std::sort(p.begin(), p.end());
    for(int i = 1; i < p.size(); ++i)
    {
        if(p[i].first != p[i - 1].first * p[i - 1].second)
            return false;
    }
    return true;
}

bool checkContiguous(const TensorDescriptor& x)
{
    size_t s = 1;
    for(int i = x.GetNumDims() - 1; i >= 0; --i)
    {
        if(s != x.GetStrides()[i])
            return false;
        s *= x.GetLengths()[i];
    }
    return true;
}

NetworkConfig L1LossFwdProblemDescription::MakeNetworkConfig() const
{
    auto input_dtype  = iDesc.GetType();
    auto output_dtype = oDesc.GetType();
    auto size         = iDesc.GetElementSize();

    std::ostringstream ss;

    ss << "l1loss_fwd";
    ss << "reduction" << reduction;
    ss << "i_dtype" << input_dtype;
    ss << "o_dtype" << output_dtype;
    ss << "size" << size;

    return NetworkConfig{ss.str()};
}

} // namespace l1loss

} // namespace miopen
