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

    friend std::ostream& operator<<(std::ostream& os, const SmoothL1LossTestCase& tc)
    {
        return os << " N:" << tc.N << " C:" << tc.C << " D:" << tc.D << " H:" << tc.H
                  << " W:" << tc.W << " Reduction:" << tc.reduction << " Beta:" << tc.beta;
    }

    std::vector<size_t> GetInput()
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
};

inline std::vector<SmoothL1LossTestCase> SmoothL1LossTestConfigs(const size_t ntest_permode)
{
    std::vector<SmoothL1LossTestCase> tcs;
    for(size_t i = 0; i < ntest_permode; ++i)
    {
        auto N    = prng::gen_A_to_B(1, 50);
        auto C    = prng::gen_A_to_B(0, 100);
        auto D    = prng::gen_A_to_B(0, 100);
        auto H    = prng::gen_A_to_B(0, 100);
        auto W    = prng::gen_A_to_B(0, 100);
        auto beta = prng::gen_A_to_B(0.0f, 4.0f);
        tcs.push_back({N, C, D, H, W, MIOPEN_LOSS_NO_REDUCTION, beta});
        // tcs.push_back({N, C, D, H, W, MIOPEN_LOSS_MEAN_REDUCTION, beta});
        // tcs.push_back({N, C, D, H, W, MIOPEN_LOSS_SUM_REDUCTION, beta});
    }
    return tcs;
}

template <typename T = float>
struct SmoothL1LossTest : public ::testing::TestWithParam<SmoothL1LossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle  = get_handle();
        smooth_l1loss_config     = GetParam();
        auto gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };

        reduction = smooth_l1loss_config.reduction;
        beta      = smooth_l1loss_config.beta;

        auto dims = smooth_l1loss_config.GetInput();

        input = tensor<T>{dims}.generate(gen_value);
        target = tensor<T>{dims}.generate(gen_value);

        output = tensor<T>{dims};
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>{dims};
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        std::vector<size_t> workspace_dims;
        ws_sizeInBytes = miopen::GetSmoothL1LossWorkspaceSize(handle, reduction, input.desc, target.desc, output.desc);
        if(ws_sizeInBytes > 0)
        {
            workspace = tensor<T>{dims};
            std::fill(workspace.begin(), workspace.end(), std::numeric_limits<T>::quiet_NaN());
            workspace_dev = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        target_dev  = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }
    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_smooth_l1loss_forward<T>(input, target, ref_output, reduction, beta);
        miopenStatus_t status;

        status = miopen::SmoothL1LossForward(handle,
                                             reduction,
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
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_output, output);

        EXPECT_TRUE(miopen::range_distance(ref_output) == miopen::range_distance(output));
        EXPECT_TRUE(error < threshold * 10) << "Error output beyond tolerance Error:" << error
                                            << ",  Thresholdx10: " << threshold * 10;
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
