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

#include "sigmoid_focal_loss.hpp"
#include <miopen/bfloat16.hpp>

namespace sigmoidfocalloss {
using GPU_SigmoidFocalLoss_fwd_FP32  = SigmoidFocalLossFwdTest<float>;
using GPU_SigmoidFocalLoss_fwd_FP16  = SigmoidFocalLossFwdTest<half>;
using GPU_SigmoidFocalLoss_fwd_BFP16 = SigmoidFocalLossFwdTest<bfloat16>;
using GPU_SigmoidFocalLoss_bwd_FP32  = SigmoidFocalLossBwdTest<float>;
using GPU_SigmoidFocalLoss_bwd_FP16  = SigmoidFocalLossBwdTest<half>;
using GPU_SigmoidFocalLoss_bwd_BFP16 = SigmoidFocalLossBwdTest<bfloat16>;
}; // namespace sigmoidfocalloss

using namespace sigmoidfocalloss;

TEST_P(GPU_SigmoidFocalLoss_fwd_FP32, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SigmoidFocalLoss_fwd_FP32,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(GPU_SigmoidFocalLoss_fwd_FP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SigmoidFocalLoss_fwd_FP16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(GPU_SigmoidFocalLoss_fwd_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SigmoidFocalLoss_fwd_BFP16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(GPU_SigmoidFocalLoss_bwd_FP32, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SigmoidFocalLoss_bwd_FP32,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(GPU_SigmoidFocalLoss_bwd_FP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SigmoidFocalLoss_bwd_FP16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));

TEST_P(GPU_SigmoidFocalLoss_bwd_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SigmoidFocalLoss_bwd_BFP16,
                         testing::ValuesIn(SigmoidFocalLossTestConfigs()));
