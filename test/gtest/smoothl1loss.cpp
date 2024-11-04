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

#include "smoothl1loss.hpp"
#include <miopen/bfloat16.hpp>

namespace smoothl1loss {

struct GPU_SmoothL1LossForward_FP32 : SmoothL1LossTestForward<float>
{
};

struct GPU_SmoothL1LossForward_FP16 : SmoothL1LossTestForward<half>
{
};

struct GPU_SmoothL1LossForward_BFP16 : SmoothL1LossTestForward<bfloat16>
{
};

struct GPU_SmoothL1LossBackward_FP32 : SmoothL1LossTestBackward<float>
{
};

struct GPU_SmoothL1LossBackward_FP16 : SmoothL1LossTestBackward<half>
{
};

struct GPU_SmoothL1LossBackward_BFP16 : SmoothL1LossTestBackward<bfloat16>
{
};

} // namespace smoothl1loss
using namespace smoothl1loss;

TEST_P(GPU_SmoothL1LossForward_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_SmoothL1LossForward_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_SmoothL1LossForward_BFP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_SmoothL1LossBackward_FP32, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_SmoothL1LossBackward_FP16, Test)
{
    RunTest();
    Verify();
};

TEST_P(GPU_SmoothL1LossBackward_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_SmoothL1LossForward_FP32,
                         testing::ValuesIn(SmoothL1LossSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_SmoothL1LossForward_FP16,
                         testing::ValuesIn(SmoothL1LossSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_SmoothL1LossForward_BFP16,
                         testing::ValuesIn(SmoothL1LossSmokeTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_SmoothL1LossForward_FP32,
                         testing::ValuesIn(SmoothL1LossPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_SmoothL1LossForward_FP16,
                         testing::ValuesIn(SmoothL1LossPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_SmoothL1LossForward_BFP16,
                         testing::ValuesIn(SmoothL1LossPerfTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SmoothL1LossForward_FP32,
                         testing::ValuesIn(SmoothL1LossFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SmoothL1LossForward_FP16,
                         testing::ValuesIn(SmoothL1LossFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SmoothL1LossForward_BFP16,
                         testing::ValuesIn(SmoothL1LossFullTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_SmoothL1LossBackward_FP32,
                         testing::ValuesIn(SmoothL1LossSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_SmoothL1LossBackward_FP16,
                         testing::ValuesIn(SmoothL1LossSmokeTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_SmoothL1LossBackward_BFP16,
                         testing::ValuesIn(SmoothL1LossSmokeTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_SmoothL1LossBackward_FP32,
                         testing::ValuesIn(SmoothL1LossPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_SmoothL1LossBackward_FP16,
                         testing::ValuesIn(SmoothL1LossPerfTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Perf,
                         GPU_SmoothL1LossBackward_BFP16,
                         testing::ValuesIn(SmoothL1LossPerfTestConfigs()));

INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SmoothL1LossBackward_FP32,
                         testing::ValuesIn(SmoothL1LossFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SmoothL1LossBackward_FP16,
                         testing::ValuesIn(SmoothL1LossFullTestConfigs()));
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_SmoothL1LossBackward_BFP16,
                         testing::ValuesIn(SmoothL1LossFullTestConfigs()));
