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

#include "smooth_l1loss.hpp"
#include <miopen/bfloat16.hpp>
#include <miopen/env.hpp>

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_TEST_FLOAT_ARG)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_TEST_ALL)

namespace smoothl1loss {

std::string GetFloatArg()
{
    const auto& tmp = miopen::GetStringEnv(ENV(MIOPEN_TEST_FLOAT_ARG));
    if(tmp.empty())
    {
        return "";
    }
    return tmp;
}

struct SmoothL1LossTestForwardFloat : SmoothL1LossTestForward<float>
{
};

struct SmoothL1LossTestForwardHalf : SmoothL1LossTestForward<half>
{
};

struct SmoothL1LossTestForwardBfloat16 : SmoothL1LossTestForward<bfloat16>
{
};

struct SmoothL1LossTestBackwardFloat : SmoothL1LossTestBackward<float>
{
};

struct SmoothL1LossTestBackwardHalf : SmoothL1LossTestBackward<half>
{
};

struct SmoothL1LossTestBackwardBfloat16 : SmoothL1LossTestBackward<bfloat16>
{
};

} // namespace smoothl1loss
using namespace smoothl1loss;

TEST_P(SmoothL1LossTestForwardFloat, SmoothL1LossTestFw)
{
    RunTest();
    Verify();
};

TEST_P(SmoothL1LossTestForwardHalf, SmoothL1LossTestFw)
{
    RunTest();
    Verify();
};

TEST_P(SmoothL1LossTestForwardBfloat16, SmoothL1LossTestFw)
{
    RunTest();
    Verify();
};

TEST_P(SmoothL1LossTestBackwardFloat, SmoothL1LossTestBw)
{
    RunTest();
    Verify();
};

TEST_P(SmoothL1LossTestBackwardHalf, SmoothL1LossTestBw)
{
    RunTest();
    Verify();
};

TEST_P(SmoothL1LossTestBackwardBfloat16, SmoothL1LossTestBw)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(SmoothL1LossTestSet,
                         SmoothL1LossTestForwardFloat,
                         testing::ValuesIn(SmoothL1LossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SmoothL1LossTestSet,
                         SmoothL1LossTestForwardHalf,
                         testing::ValuesIn(SmoothL1LossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SmoothL1LossTestSet,
                         SmoothL1LossTestForwardBfloat16,
                         testing::ValuesIn(SmoothL1LossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SmoothL1LossTestSet,
                         SmoothL1LossTestBackwardFloat,
                         testing::ValuesIn(SmoothL1LossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SmoothL1LossTestSet,
                         SmoothL1LossTestBackwardHalf,
                         testing::ValuesIn(SmoothL1LossTestConfigs()));
INSTANTIATE_TEST_SUITE_P(SmoothL1LossTestSet,
                         SmoothL1LossTestBackwardBfloat16,
                         testing::ValuesIn(SmoothL1LossTestConfigs()));
