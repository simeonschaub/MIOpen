/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <miopen/bz2.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include <bzlib.h>

namespace miopen {
void check_bz2_error(int e, const std::string& name)
{
    if(e == BZ_OK)
        return;
    if(e == BZ_MEM_ERROR)
        throw std::runtime_error(name + " failed: out of memory!");
    if(e == BZ_OUTBUFF_FULL)
    {
        throw std::runtime_error(name +
                                 " failed: the size of the compressed data exceeds *destLen");
    }
    if(e == BZ_PARAM_ERROR)
        throw std::runtime_error(name + " failed: bad parameters given to function");
    if(e == BZ_DATA_ERROR)
    {
        throw std::runtime_error(
            name + " failed: a data integrity error was detected in the compressed data");
    }
    if(e == BZ_DATA_ERROR_MAGIC)
    {
        throw std::runtime_error(
            name + " failed: the compressed data doesn't begin with the right magic bytes");
    }
    if(e == BZ_UNEXPECTED_EOF)
        throw std::runtime_error(name + " failed: the compressed data ends unexpectedly");
    throw std::runtime_error(name + " failed: unknown error!");
}

std::vector<char> compress(const std::vector<char>& v, bool* compressed)
{
    auto result      = v;
    unsigned int len = result.size();
    // NOLINTBEGIN(cppcoreguidelines-pro-type-const-cast)
    auto e = BZ2_bzBuffToBuffCompress(
        result.data(), &len, const_cast<char*>(v.data()), v.size(), 9, 0, 30);
    // NOLINTEND(cppcoreguidelines-pro-type-const-cast)
    if(compressed != nullptr and e == BZ_OUTBUFF_FULL)
    {
        *compressed = false;
        return v;
    }
    check_bz2_error(e, "BZ2_bzBuffToBuffCompress");
    result.resize(len);
    if(compressed != nullptr)
        *compressed = true;
    return result;
}

std::vector<char> decompress(const std::vector<char>& v, unsigned int size)
{
    std::vector<char> result(size, 0);
    unsigned int len = result.size();
    // NOLINTBEGIN(cppcoreguidelines-pro-type-const-cast)
    auto e = BZ2_bzBuffToBuffDecompress(
        result.data(), &len, const_cast<char*>(v.data()), v.size(), 0, 0);
    // NOLINTEND(cppcoreguidelines-pro-type-const-cast)
    check_bz2_error(e, "BZ2_bzBuffToBuffDecompress");
    result.resize(len);
    return result;
}

} // namespace miopen
