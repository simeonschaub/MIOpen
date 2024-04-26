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

struct SmoothL1LossTestCase
{
    size_t N;
    size_t C;
    size_t D;
    size_t H;
    size_t W;
    miopenLossReduction_t reduction;
    float beta;
    std::string permutation;

    friend std::ostream& operator<<(std::ostream& os, const SmoothL1LossTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " Reduction:" << tc.reduction << " Beta:" << tc.beta
                  << " Permutation:\"" << tc.permutation << "\"";
    }

    std::vector<size_t> GetInputDims()
    {
        if(beta < 0)
        {
            std::cout << "Error Beta Value\n" << std::endl;
        }
        if((N != 0) && (C != 0) && (D != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, D, H, W});
        }
        else if((N != 0) && (C != 0) && (H != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, H, W});
        }
        else if((N != 0) && (C != 0) && (W != 0))
        {
            return std::vector<size_t>({N, C, W});
        }
        else if((N != 0) && (W != 0))
        {
            return std::vector<size_t>({N, W});
        }
        else if((N != 0))
        {
            return std::vector<size_t>({N});
        }
        else
        {
            std::cout << "Error Input Tensor Lengths\n" << std::endl;
            return std::vector<size_t>({0});
        }
    }

    std::vector<size_t> GetInputStrides()
    {
        if(permutation.empty())
            return {};
        auto dims = GetInputDims();
        std::vector<size_t> strides(dims.size());
        size_t cur_stride = 1;
        for(auto c : permutation)
        {
            int idx = c - '0';
            if(idx < 0 || dims.size() <= idx)
            {
                cur_stride = 0;
                break;
            }
            strides[idx] = cur_stride;
            cur_stride *= dims[idx];
            dims[idx] = 0;
        }
        if(cur_stride == 0)
        {
            std::cout << "Error Input Tensor Shape Permutations\n" << std::endl;
            return std::vector<size_t>(dims.size(), 0);
        }
        return strides;
    }
};

inline std::vector<SmoothL1LossTestCase> SmoothL1LossUnreducedForwardContiguousTestConfigs()
{
    std::vector<SmoothL1LossTestCase> tcs;
    tcs.push_back({1, 1, 0, 2, 2, MIOPEN_LOSS_NO_REDUCTION, 1, ""});
    tcs.push_back({2, 10, 0, 128, 128, MIOPEN_LOSS_NO_REDUCTION, 1, ""});
    tcs.push_back({5, 13, 0, 17, 11, MIOPEN_LOSS_NO_REDUCTION, 1, ""});
    tcs.push_back({256, 4, 0, 0, 8723, MIOPEN_LOSS_NO_REDUCTION, 1, ""});
    return tcs;
}

inline std::vector<SmoothL1LossTestCase> SmoothL1LossUnreducedForwardTestConfigs()
{
    std::vector<SmoothL1LossTestCase> tcs;
    tcs.push_back({1, 1, 0, 2, 2, MIOPEN_LOSS_NO_REDUCTION, 1, "1320"});
    tcs.push_back({2, 10, 0, 128, 128, MIOPEN_LOSS_NO_REDUCTION, 1, "0231"});
    tcs.push_back({5, 13, 0, 17, 11, MIOPEN_LOSS_NO_REDUCTION, 1, "2013"});
    tcs.push_back({256, 4, 0, 0, 8723, MIOPEN_LOSS_NO_REDUCTION, 1, "102"});
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

template <typename T = float>
struct SmoothL1LossTest : public ::testing::TestWithParam<SmoothL1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle        = get_handle();
        smooth_l1loss_config = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        reduction = smooth_l1loss_config.reduction;
        beta      = smooth_l1loss_config.beta;

        auto dims    = smooth_l1loss_config.GetInputDims();
        auto strides = smooth_l1loss_config.GetInputStrides();

        if(strides.empty())
        {
            input      = tensor<T>{dims}.generate(gen_value);
            target     = tensor<T>{dims}.generate(gen_value);
            output     = tensor<T>{dims};
            ref_output = tensor<T>{dims};
        }
        else
        {
            input      = tensor<T>{dims, strides}.generate(gen_value);
            target     = tensor<T>{dims, strides}.generate(gen_value);
            output     = tensor<T>{dims, strides};
            ref_output = tensor<T>{dims, strides};
        }
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_dims;
        ws_sizeInBytes = miopen::GetSmoothL1LossWorkspaceSize(
            handle, reduction, input.desc, target.desc, output.desc);
        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<T>{dims};
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

        cpu_smooth_l1loss_forward<T>(input, target, ref_output, reduction, beta);
        miopenStatus_t status;

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
