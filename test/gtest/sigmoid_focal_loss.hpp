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
#include "cpu_sigmoid_focal_loss.hpp"
#include "get_handle.hpp"
#include "miopen/allocator.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/sigmoid_focal_loss.hpp>

struct SigmoidFocalLossTestCase
{
    std::vector<size_t> dims;
    float alpha;
    float gamma;
    miopenLossReductionMode_t reduction;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const SigmoidFocalLossTestCase& tc)
    {
        os << "dims: ";
        for(auto dim : tc.dims)
        {
            os << dim << " ";
        }
        return os << "is_contiguous: " << tc.isContiguous << " alpha: " << tc.alpha
                  << " gamma: " << tc.gamma << " reduction: " << tc.reduction;
    }
};

inline std::vector<size_t> ComputeStrides(std::vector<size_t> dims, const bool isContiguous)
{
    if(!isContiguous)
        std::swap(dims.front(), dims.back());
    std::vector<size_t> strides(dims.size());
    strides.back() = 1;
    for(int i = dims.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * dims[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

inline std::vector<SigmoidFocalLossTestCase> SigmoidFocalLossTestConfigs()
{
    std::vector<std::vector<size_t>> dimss = {
        {1},
        {4000},
        {64},
        {100, 500},
        {10, 20, 200},
        {8, 3, 20, 100},
        {2, 2, 3, 4, 100},
        {10},
    };

    std::vector<SigmoidFocalLossTestCase> tcs;
    for(auto reduction :
        {MIOPEN_LOSS_REDUCTION_NONE, MIOPEN_LOSS_REDUCTION_SUM, MIOPEN_LOSS_REDUCTION_MEAN})
        for(auto contiguous : {true, false})
            for(const auto& dims : dimss)
                tcs.push_back({dims, 0.25, 2, reduction, contiguous});

    return tcs;
}

template <typename T>
struct SigmoidFocalLossFwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        reduction = config.reduction;

        auto in_dims    = config.dims;
        auto in_strides = ComputeStrides(in_dims, config.isContiguous);

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 1); };
        input             = tensor<T>{in_dims, in_strides}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 2); };
        target             = tensor<T>{in_dims, in_strides}.generate(tar_gen_value);

        auto out_dims =
            (reduction == MIOPEN_LOSS_REDUCTION_NONE ? in_dims : std::vector<size_t>{1});

        output = tensor<T>(out_dims);
        std::fill(output.begin(), output.end(), std::numeric_limits<T>::quiet_NaN());

        ref_output = tensor<T>(out_dims);
        std::fill(ref_output.begin(), ref_output.end(), std::numeric_limits<T>::quiet_NaN());

        size_t workspaceSizeBytes = miopen::GetSigmoidFocalLossForwardWorkspaceSize(
            handle, input.desc, target.desc, output.desc, reduction);
        if(workspaceSizeBytes == static_cast<size_t>(-1))
            GTEST_SKIP();

        if(workspaceSizeBytes != 0)
        {
            size_t workspaceElements = workspaceSizeBytes / sizeof(float);
            workspace                = tensor<float>(workspaceElements);
            workspace_dev            = handle.Write(workspace.data);
        }

        input_dev  = handle.Write(input.data);
        target_dev = handle.Write(target.data);
        output_dev = handle.Write(output.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::SigmoidFocalLossForward(handle,
                                                 workspace_dev.get(),
                                                 workspace.GetDataByteSize(),
                                                 input.desc,
                                                 input_dev.get(),
                                                 target.desc,
                                                 target_dev.get(),
                                                 output.desc,
                                                 output_dev.get(),
                                                 config.alpha,
                                                 config.gamma,
                                                 reduction);
        cpu_sigmoid_focal_loss_forward<T>(
            input, target, ref_output, config.alpha, config.gamma, reduction);

        EXPECT_EQ(status, miopenStatusSuccess);

        output.data = handle.Read<T>(output_dev, output.data.size());
    }

    void Verify()
    {
        double threshold = get_tolerance<T>();

        auto error = miopen::rms_range(ref_output, output);

        ASSERT_EQ(miopen::range_distance(ref_output), miopen::range_distance(output));
        EXPECT_LT(error, threshold)
            << "Error output beyond tolerance Error: " << error << ",  Threshold: " << threshold;
    }
    SigmoidFocalLossTestCase config;
    miopenLossReductionMode_t reduction;

    tensor<T> input;
    tensor<T> target;
    tensor<T> output;
    tensor<float> workspace;

    tensor<T> ref_output;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr workspace_dev;
    miopen::Allocator::ManageDataPtr output_dev;
};

template <typename T>
struct SigmoidFocalLossBwdTest : public ::testing::TestWithParam<SigmoidFocalLossTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle = get_handle();
        config        = GetParam();

        reduction = config.reduction;

        auto in_dims    = config.dims;
        auto in_strides = ComputeStrides(in_dims, config.isContiguous);

        auto in_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 1); };
        input             = tensor<T>{in_dims, in_strides}.generate(in_gen_value);

        auto tar_gen_value = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 2); };
        target             = tensor<T>{in_dims, in_strides}.generate(tar_gen_value);

        auto out_dims =
            (reduction == MIOPEN_LOSS_REDUCTION_NONE ? in_dims : std::vector<size_t>{1});

        dOutput = tensor<T>(out_dims);
        std::fill(dOutput.begin(), dOutput.end(), 1);

        dInput = tensor<T>{in_dims};
        std::fill(dInput.begin(), dInput.end(), 0);

        ref_dInput = tensor<T>{in_dims};
        std::fill(ref_dInput.begin(), ref_dInput.end(), 0);

        dTarget = tensor<T>{in_dims};
        std::fill(dTarget.begin(), dTarget.end(), 0);

        ref_dTarget = tensor<T>{in_dims};
        std::fill(ref_dTarget.begin(), ref_dTarget.end(), 0);

        input_dev   = handle.Write(input.data);
        target_dev  = handle.Write(target.data);
        dOutput_dev = handle.Write(dOutput.data);
        dInput_dev  = handle.Write(dInput.data);
        dTarget_dev = handle.Write(dTarget.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        miopenStatus_t status;

        status = miopen::SigmoidFocalLossBackward(handle,
                                                  input.desc,
                                                  input_dev.get(),
                                                  target.desc,
                                                  target_dev.get(),
                                                  dOutput.desc,
                                                  dOutput_dev.get(),
                                                  dInput.desc,
                                                  dInput_dev.get(),
                                                  dTarget.desc,
                                                  dTarget_dev.get(),
                                                  config.alpha,
                                                  config.gamma,
                                                  reduction);
        cpu_sigmoid_focal_loss_backward<T>(
            input, target, dOutput, ref_dInput, ref_dTarget, config.alpha, config.gamma, reduction);

        EXPECT_EQ(status, miopenStatusSuccess);

        dInput.data  = handle.Read<T>(dInput_dev, dInput.data.size());
        dTarget.data = handle.Read<T>(dTarget_dev, dTarget.data.size());
    }

    void Verify()
    {
        double threshold = get_tolerance<T>();

        auto dInputError = miopen::rms_range(ref_dInput, dInput);

        ASSERT_EQ(miopen::range_distance(ref_dInput), miopen::range_distance(dInput));
        EXPECT_LT(dInputError, threshold)
            << "dInput error output beyond tolerance Error: " << dInputError
            << ",  Threshold: " << threshold;

        auto dTargetError = miopen::rms_range(ref_dTarget, dTarget);

        ASSERT_EQ(miopen::range_distance(ref_dTarget), miopen::range_distance(dTarget));
        EXPECT_LT(dTargetError, threshold)
            << "dTarget error output beyond tolerance Error: " << dTargetError
            << ",  Threshold: " << threshold;
    }
    SigmoidFocalLossTestCase config;
    miopenLossReductionMode_t reduction;

    tensor<T> input;
    tensor<T> target;
    tensor<T> dOutput;
    tensor<T> dInput;
    tensor<T> dTarget;

    tensor<T> ref_dInput;
    tensor<T> ref_dTarget;

    miopen::Allocator::ManageDataPtr input_dev;
    miopen::Allocator::ManageDataPtr target_dev;
    miopen::Allocator::ManageDataPtr dOutput_dev;
    miopen::Allocator::ManageDataPtr dInput_dev;
    miopen::Allocator::ManageDataPtr dTarget_dev;
};
