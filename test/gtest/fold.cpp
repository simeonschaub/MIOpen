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

#include "fold.hpp"

namespace fold {

struct GPU_Fold_fwd_FP32 : FoldFwdTest<float>
{
};

struct GPU_Fold_fwd_FP16 : FoldFwdTest<half>
{
};

struct GPU_Fold_fwd_BFP16 : FoldFwdTest<bfloat16>
{
};

struct GPU_Fold_bwd_FP32 : FoldBwdTest<float>
{
};

struct GPU_Fold_bwd_FP16 : FoldBwdTest<half>
{
};

struct GPU_Fold_bwd_BFP16 : FoldBwdTest<bfloat16>
{
};
}; // namespace fold

using namespace fold;

TEST_P(GPU_Fold_fwd_FP32, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fold_fwd_FP32, testing::ValuesIn(FoldTestConfigs()));

TEST_P(GPU_Fold_fwd_FP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fold_fwd_FP16, testing::ValuesIn(FoldTestConfigs()));

TEST_P(GPU_Fold_fwd_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fold_fwd_BFP16, testing::ValuesIn(FoldTestConfigs()));

TEST_P(GPU_Fold_bwd_FP32, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fold_bwd_FP32, testing::ValuesIn(FoldTestConfigs()));

TEST_P(GPU_Fold_bwd_FP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fold_bwd_FP16, testing::ValuesIn(FoldTestConfigs()));

TEST_P(GPU_Fold_bwd_BFP16, Test)
{
    RunTest();
    Verify();
};

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Fold_bwd_BFP16, testing::ValuesIn(FoldTestConfigs()));
