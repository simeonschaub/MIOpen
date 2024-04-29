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

#include "../driver/tensor_driver.hpp"
#include "cpu_smooth_l1loss.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/smooth_l1loss.hpp>

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

struct SmoothL1LossTestCase
{
    std::vector<size_t> lengths;
    miopenLossReduction_t reduction;
    float beta;
    bool contiguous;

    friend std::ostream& operator<<(std::ostream& os, const SmoothL1LossTestCase& tc)
    {
        return os << " Lengths:" << tc.lengths << " Reduction:" << tc.reduction
                  << " Beta:" << tc.beta << " Contiguous:" << (tc.contiguous ? "True" : "False");
    }
};

inline std::vector<SmoothL1LossTestCase> SmoothL1LossUnreducedForwardContiguousTestConfigs()
{
    std::vector<SmoothL1LossTestCase> tcs;
    tcs.push_back({{1, 1, 2, 2}, MIOPEN_LOSS_NO_REDUCTION, 1, true});
    tcs.push_back({{2, 10, 128, 128}, MIOPEN_LOSS_NO_REDUCTION, 1, true});
    tcs.push_back({{5, 13, 17, 11}, MIOPEN_LOSS_NO_REDUCTION, 1, true});
    tcs.push_back({{256, 4, 8723}, MIOPEN_LOSS_NO_REDUCTION, 1, true});
    return tcs;
}

inline std::vector<SmoothL1LossTestCase> SmoothL1LossUnreducedForwardTestConfigs()
{
    std::vector<SmoothL1LossTestCase> tcs;
    tcs.push_back({{1, 1, 2, 2}, MIOPEN_LOSS_NO_REDUCTION, 1, false});
    tcs.push_back({{2, 10, 128, 128}, MIOPEN_LOSS_NO_REDUCTION, 1, false});
    tcs.push_back({{5, 13, 17, 11}, MIOPEN_LOSS_NO_REDUCTION, 1, false});
    tcs.push_back({{256, 4, 8723}, MIOPEN_LOSS_NO_REDUCTION, 1, false});
    return tcs;
}

inline std::vector<SmoothL1LossTestCase> SmoothL1LossTestConfigs()
{
    std::vector<SmoothL1LossTestCase> tcs, temp;
    temp = SmoothL1LossUnreducedForwardContiguousTestConfigs();
    tcs.insert(tcs.end(), temp.begin(), temp.end());
    temp = SmoothL1LossUnreducedForwardTestConfigs();
    tcs.insert(tcs.end(), temp.begin(), temp.end());
    return tcs;
}

inline std::vector<size_t> GetStrides(std::vector<size_t> lengths, bool contiguous)
{
    if(!contiguous)
        std::swap(lengths.front(), lengths.back());
    std::vector<size_t> strides(lengths.size());
    strides.back() = 1;
    for(int i = lengths.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * lengths[i + 1];
    if(!contiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename T = float>
struct SmoothL1LossTest : public ::testing::TestWithParam<SmoothL1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        smooth_l1loss_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        reduction       = smooth_l1loss_config.reduction;
        beta            = smooth_l1loss_config.beta;
        auto lengths    = smooth_l1loss_config.lengths;
        auto contiguous = smooth_l1loss_config.contiguous;

        auto in_strides = GetStrides(lengths, true);
        input           = tensor<T>{lengths, in_strides}.generate(gen_value);

        auto tar_strides = GetStrides(lengths, contiguous);
        target           = tensor<T>{lengths, tar_strides}.generate(gen_value);

        output = tensor<T>{lengths, in_strides};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{lengths, in_strides};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_lengths;
        ws_sizeInBytes = miopen::GetSmoothL1LossWorkspaceSize(
            handle, reduction, input.desc, target.desc, output.desc);
        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<T>{lengths};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());
            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        if(output.desc.GetElementSize() > 1) // unreduced cases
        {
            cpu_smooth_l1loss_unreduced_forward<T>(input, target, ref_output, beta);
            status = miopen::SmoothL1LossUnreducedForward(handle,
                                                          input.desc,
                                                          input_dev.get(),
                                                          target.desc,
                                                          target_dev.get(),
                                                          output.desc,
                                                          output_dev.get(),
                                                          beta);
        }
        else // reduced cases
        {
            cpu_smooth_l1loss_reduced_forward<T>(input, target, ref_output, beta, 1);
            status = miopen::SmoothL1LossForward(handle,
                                                 reduction,
                                                 nullptr,
                                                 0,
                                                 input.desc,
                                                 input_dev.get(),
                                                 target.desc,
                                                 target_dev.get(),
                                                 output.desc,
                                                 output_dev.get(),
                                                 beta);
        }

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        // Computation error of fp16 is ~2^13 (=8192) bigger than
        // the one of fp32 because mantissa is shorter by 13 bits.
        double tolerance = std::is_same<T, float>::value ? 1.5e-6 : 8.2e-3;

        // bf16 mantissa has 7 bits, by 3 bits shorter than fp16.
        if(std::is_same<T, bfloat16>::value)
            tolerance *= 8.0;

        auto error = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < tolerance)
            << "Error output beyond tolerance Error:" << error << ",  Tolerance: " << tolerance;
    }
    SmoothL1LossTestCase smooth_l1loss_config;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<T> workspace;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr output_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;

    size_t ws_sizeInBytes;

    miopenLossReduction_t reduction;
    float beta;
};
